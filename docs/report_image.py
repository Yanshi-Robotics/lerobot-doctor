#!/usr/bin/env python3
"""Render docs/images/report.png from an example report, without running the doctor.

    python3 docs/report_image.py                                   # both boxes -> docs/images/report.png
    python3 docs/report_image.py --lang zh --out /some/where.png   # one box only (a PDF page fits one)
    python3 docs/report_image.py --example examples/<other>.json

The boxes are drawn exactly as the terminal would print them, colours included, into an HTML page
where every character sits on a fixed cell grid (CJK = two cells), then headless Chrome takes the
screenshot and Pillow crops it. Re-run after any change to the report.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = 100                    # the report caps itself at REPORT_MAX_WIDTH = 100 cells
CELL_PX, LINE_PX, FONT_PX, PAD_PX = 8, 18, 13.3, 20
COLORS = {"31": "#e06c75", "32": "#98c379", "33": "#e5c07b"}
_ANSI = re.compile(r"\x1b\[([0-9;]*)m")


def load_doctor():
    spec = importlib.util.spec_from_file_location("lerobot_doctor", ROOT / "lerobot_doctor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_text(doc, example: Path, lang: str | None) -> str:
    report = json.loads(example.read_text(encoding="utf-8"))
    os.environ["COLUMNS"] = str(COLUMNS + 1)          # render_report uses columns - 1
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    con = doc.Console(stream=stream)
    con.tty = True                                     # colours on, as on a real terminal
    verdicts = doc.evaluate(report)
    if lang:
        doc.render_one(con, report, verdicts, lang, COLUMNS)
    else:
        # the path line as a user sees it: the run's dated file, derived from the example's own timestamp
        stamp = (report.get("started_at") or "2026-09-20T00:00")[:16].replace("T", "-").replace(":", "")
        doc.render_report(con, report, verdicts, Path("~") / "lerobot-doctor" / f"report-{stamp}.json")
    stream.seek(0)
    return stream.read()


def to_html(text: str) -> str:
    def cell(ch: str) -> str:
        wide = unicodedata.east_asian_width(ch) in ("W", "F")
        return f'<i class="{"w" if wide else "n"}">{html.escape(ch)}</i>'

    out, style = [], {"color": None, "bold": False, "dim": False}
    for line in text.strip("\n").splitlines():
        pos, pieces = 0, []
        for m in _ANSI.finditer(line):
            pieces.append((dict(style), line[pos:m.start()]))
            for code in (m.group(1) or "0").split(";"):
                if code == "0":
                    style = {"color": None, "bold": False, "dim": False}
                elif code == "1":
                    style["bold"] = True
                elif code == "2":
                    style["dim"] = True
                elif code in COLORS:
                    style["color"] = COLORS[code]
            pos = m.end()
        pieces.append((dict(style), line[pos:]))
        row = []
        for st, seg in pieces:
            if not seg:
                continue
            css = ";".join(filter(None, [f"color:{st['color']}" if st["color"] else "", "font-weight:700" if st["bold"] else "",
                                         "opacity:.55" if st["dim"] else ""]))
            row.append(f'<span style="{css}">{"".join(cell(c) for c in seg)}</span>')
        out.append("".join(row) or "&nbsp;")
    lines = "".join(f"<div>{r}</div>" for r in out)     # no newlines: the container is white-space:pre
    return f"""<!doctype html><meta charset="utf-8"><style>
body{{margin:0;background:#1b1d23}}
#t{{display:inline-block;padding:{PAD_PX}px;background:#1b1d23;color:#d8dee9;font:{FONT_PX}px/{LINE_PX}px "DejaVu Sans Mono","Noto Sans CJK SC",monospace;white-space:pre}}
#t div{{height:{LINE_PX}px;white-space:nowrap}}
#t i{{font-style:normal;display:inline-block;text-align:center;overflow:hidden;vertical-align:top}}
#t i.n{{width:{CELL_PX}px}} #t i.w{{width:{2 * CELL_PX}px}}
</style><div id="t">{lines}</div>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", type=Path, default=ROOT / "examples" / "linux-ubuntu24-rtx5070ti-16gb.json")
    ap.add_argument("--lang", choices=["en", "zh"], default=None, help="one box only; default draws both")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "report.png")
    a = ap.parse_args()
    doc = load_doctor()
    text = render_text(doc, a.example, a.lang)
    n_lines = len(text.strip("\n").splitlines())
    width_px = COLUMNS * CELL_PX + 2 * PAD_PX + 120
    height_px = n_lines * LINE_PX + 2 * PAD_PX + 40
    chrome = next((c for c in ("google-chrome", "google-chrome-stable", "chromium") if subprocess.run(["which", c], capture_output=True).returncode == 0), None)
    if not chrome:
        print("no chrome", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(to_html(text), encoding="utf-8")
        shot = Path(tmp) / "shot.png"
        subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={width_px},{height_px}", f"--screenshot={shot}", f"file://{page}"],
                       check=True, capture_output=True, timeout=120)
        from PIL import Image, ImageChops
        im = Image.open(shot).convert("RGB")
        bg = Image.new("RGB", im.size, (0x1b, 0x1d, 0x23))
        box = ImageChops.difference(im, bg).getbbox()      # the drawn text; the panel colour equals the page colour
        m = 2 * PAD_PX
        im = im.crop((max(box[0] - m, 0), max(box[1] - m, 0), min(box[2] + m, im.width), min(box[3] + m, im.height)))
        im.save(a.out, optimize=True)
    print(f"wrote {a.out} {im.size[0]}x{im.size[1]} from {a.example.name} ({n_lines} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
