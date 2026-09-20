#!/usr/bin/env python3
"""Check README.md and every docs/i18n/<lang>/README.md before pushing.

    python3 docs/check_readme.py

Limits follow the measured benchmarks in agent-rules/workflows/github.md:
about 150 lines, under 10 table rows, single-digit bold. Links and images are
checked against what git tracks, not the local file system.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LINES, MAX_TABLE_ROWS, MAX_BOLD = 160, 10, 9
EXTERNAL = ("http://", "https://", "#", "mailto:")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def code_blocks(text):
    """[(language, body)] for every fenced block, in order."""
    return [(lang, body) for lang, body in re.findall(r"^```(\w*)\n(.*?)^```", text, re.M | re.S)]


def check(path, tracked, failures):
    rel = path.relative_to(ROOT)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        failures.append(f"{rel}: {len(lines)} lines, limit {MAX_LINES}")
    table_rows = sum(1 for line in lines if line.startswith("|") and not re.match(r"^\|\s*-", line))
    if table_rows > MAX_TABLE_ROWS:
        failures.append(f"{rel}: {table_rows} table rows, limit {MAX_TABLE_ROWS}")
    bold = len(re.findall(r"\*\*[^*\n]+\*\*", text))
    if bold > MAX_BOLD:
        failures.append(f"{rel}: {bold} bold spans, limit {MAX_BOLD}")
    inside = False
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            if not inside and line.strip() == "```":
                failures.append(f"{rel}:{number}: fenced code block without a language tag")
            inside = not inside
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:href|src)="([^"]+)"', text)
    for target in targets:
        if target.startswith(EXTERNAL):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        try:
            inside_repo = str(resolved.relative_to(ROOT))
        except ValueError:
            failures.append(f"{rel}: link {target} leaves the repository")
            continue
        if inside_repo not in tracked:
            failures.append(f"{rel}: link {target} is not tracked by git")
    return sum(1 for line in lines if line.startswith("## "))


def main():
    tracked = tracked_files()
    readmes = [ROOT / "README.md", *sorted(ROOT.glob("docs/i18n/*/README.md"))]
    failures, headings = [], {}
    for path in readmes:
        headings[path.relative_to(ROOT)] = check(path, tracked, failures)
    counts = set(headings.values())
    if len(counts) > 1:
        failures.append("H2 counts differ across languages: " + ", ".join(f"{k}={v}" for k, v in headings.items()))
    # The one-liners and the development commands are the same bytes in every language: a translation
    # that drifts here ships a different command to one audience (the 0.1.2 launcher-tag slip).
    blocks = {path.relative_to(ROOT): code_blocks(path.read_text(encoding="utf-8")) for path in readmes}
    ref_name, ref = next(iter(blocks.items()))
    for name, mine in list(blocks.items())[1:]:
        if len(mine) != len(ref):
            failures.append(f"{name}: {len(mine)} fenced blocks, {ref_name} has {len(ref)}")
            continue
        for i, ((lang_a, body_a), (lang_b, body_b)) in enumerate(zip(ref, mine), 1):
            if lang_a in ("bash", "powershell") and (lang_a, body_a) != (lang_b, body_b):
                failures.append(f"{name}: fenced block {i} ({lang_a}) differs from {ref_name}")
    if failures:
        print("README check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"README check passed: {len(readmes)} files, {next(iter(counts))} H2 each, links tracked by git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
