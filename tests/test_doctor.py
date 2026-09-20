"""Unit tests for the pure parts of lerobot_doctor.py: spec parsers, accelerator classification,
hard floors, the ladder state machine, verdicts and the route rules. No network, no torch.

    python tests/test_doctor.py
"""

import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lerobot_doctor", ROOT / "lerobot_doctor.py")
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)

L = {lv.id: lv for lv in doc.LEVELS}


def specs(**over):
    base = {"system": "Linux", "os": "Ubuntu", "arch": "x86_64", "ram_gb": 32.0, "disk_free_gb": 200.0,
            "nvidia": [], "other_gpus": [], "rosetta": False, "macos_version": (0, 0), "wsl": False}
    base.update(over)
    base.update(doc.classify_accelerator(base))
    return base


def nvidia(vram=16.0, driver="580.65", cap=12.0):
    return [{"name": "GPU", "vram_gb": vram, "driver": driver, "compute_cap": cap}]


class ParserTests(unittest.TestCase):
    def test_nvidia_smi_rows(self):
        rows = doc.parse_nvidia_smi("NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 580.159.03, 12.0\n")
        self.assertEqual(rows[0]["name"], "NVIDIA GeForce RTX 5070 Ti")
        self.assertEqual(rows[0]["vram_gb"], 15.9)
        self.assertEqual(rows[0]["compute_cap"], 12.0)

    def test_nvidia_smi_without_compute_cap(self):
        rows = doc.parse_nvidia_smi("Tesla K80, 11441 MiB, 470.10\n")
        self.assertIsNone(rows[0]["compute_cap"])

    def test_driver_version(self):
        self.assertEqual(doc.parse_driver_version("570.86.10"), (570, 86))
        self.assertEqual(doc.parse_driver_version("garbage"), (0, 0))

    def test_meminfo_and_os_release(self):
        self.assertEqual(doc.parse_meminfo("MemTotal:       48441044 kB\nMemFree: 1 kB"), 49.6)
        self.assertEqual(doc.parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04 LTS"\n'), "Ubuntu 24.04 LTS")

    def test_windows_11_from_build(self):
        self.assertEqual(doc.windows_release_name("10", "10.0.22631"), "11")
        self.assertEqual(doc.windows_release_name("10", "10.0.19045"), "10")
        self.assertEqual(doc.windows_release_name("8.1", "6.3.9600"), "8.1")
        self.assertEqual(doc.windows_release_name("10", ""), "10")


class AcceleratorTests(unittest.TestCase):
    def test_modern_nvidia_is_cuda_bf16(self):
        s = specs(nvidia=nvidia())
        self.assertEqual((s["accelerator"], s["bf16"], s["torch_backend"]), ("cuda", True, "cu128"))
        self.assertEqual(s["device_mem_gb"], 16.0)

    def test_turing_has_no_bf16(self):
        s = specs(nvidia=nvidia(vram=8, cap=7.5))
        self.assertEqual(s["accelerator"], "cuda")
        self.assertFalse(s["bf16"])

    def test_old_driver_falls_back_to_cpu_but_is_not_a_floor(self):
        s = specs(nvidia=nvidia(driver="535.10"))
        self.assertEqual(s["accelerator"], "cpu")
        self.assertEqual(s["accelerator_note"], "driver_too_old")
        self.assertEqual(doc.hard_floors(s), [])

    def test_apple_silicon_is_mps(self):
        s = specs(system="Darwin", arch="arm64", macos_version=(14, 5), ram_gb=24.0)
        self.assertEqual((s["accelerator"], s["device_mem_gb"]), ("mps", 24.0))

    def test_rosetta_and_intel_mac_are_cpu(self):
        self.assertEqual(specs(system="Darwin", arch="arm64", rosetta=True, macos_version=(14, 0))["accelerator_note"], "rosetta")
        self.assertEqual(specs(system="Darwin", arch="x86_64", macos_version=(13, 0))["accelerator_note"], "intel_mac")

    def test_amd_only_is_cpu_with_note(self):
        s = specs(other_gpus=["AMD"])
        self.assertEqual((s["accelerator"], s["accelerator_note"]), ("cpu", "non_nvidia_gpu"))


class FloorTests(unittest.TestCase):
    def test_disk_and_ram_floors(self):
        keys = [f["key"] for f in doc.hard_floors(specs(disk_free_gb=12.0, ram_gb=6.0))]
        self.assertEqual(keys, ["disk", "ram"])

    def test_windows_7_is_a_floor_windows_11_is_not(self):
        self.assertEqual(doc.hard_floors(specs(system="Windows", windows_release="7"))[0]["key"], "windows")
        self.assertEqual(doc.hard_floors(specs(system="Windows", windows_release="11")), [])

    def test_weight_floor_uses_the_load_dtype(self):
        # lerobot materialises every checkpoint in float32 before casting, so 4.2B params need 16.9 GB
        fl = doc.weight_floor(L["L5"], params=4_224_041_072, dtype="bfloat16", device_mem_gb=15.9)
        self.assertEqual(fl["need_gb"], 16.9)
        self.assertIn("16.9 GB", fl["zh"])
        self.assertIsNone(doc.weight_floor(L["L5"], 4_224_041_072, "bfloat16", 24.0, ram_gb=64.0))
        self.assertIsNone(doc.weight_floor(L["L1"], None, "float32", 16.0))

    def test_weight_floor_checks_ram_too(self):
        fl = doc.weight_floor(L["L5"], 4_224_041_072, "bfloat16", 24.0, ram_gb=16.0)
        self.assertIn("RAM", fl["en"])


class LadderTests(unittest.TestCase):
    W = {"L3": {"params": 450e6}, "L4": {"params": 3.0e9}, "L5": {"params": 3.6e9}}

    def test_weights_floor_skips_only_that_level(self):
        s = specs(nvidia=nvidia(vram=4.0))
        self.W = {"L3": {"params": 450e6}, "L4": {"params": 0.88e9}, "L5": {"params": 4.2e9}}
        self.assertIsNone(doc.infer_precheck(L["L3"], {}, s, self.W, "bfloat16"))
        r = doc.infer_precheck(L["L5"], {}, s, self.W, "bfloat16")
        self.assertEqual((r["status"], r["evidence"], r["reason"]), ("SKIPPED_FLOOR", "floor", "weights_exceed_memory"))

    def test_oom_is_memory_monotonic(self):
        s = specs(nvidia=nvidia(vram=16.0))
        results = {"L4": {"infer": {"status": "FAIL_OOM", "dtype": "bfloat16"}}}
        r = doc.infer_precheck(L["L5"], results, s, self.W, "bfloat16")
        self.assertEqual((r["status"], r["reason"]), ("SKIPPED_FLOOR", "smaller_level_oom:L4"))

    def test_too_slow_never_skips_the_next_level(self):
        s = specs()  # cpu
        results = {"L3": {"infer": {"status": "TOO_SLOW"}}, "L4": {"infer": {"status": "TIMEOUT"}}}
        self.assertIsNone(doc.infer_precheck(L["L5"], results, s, self.W, "float32"))

    def test_crash_or_download_never_skips(self):
        s = specs(nvidia=nvidia())
        results = {"L3": {"infer": {"status": "FAIL_CRASH"}}, "L4": {"infer": {"status": "FAIL_DOWNLOAD"}}}
        self.assertIsNone(doc.infer_precheck(L["L5"], results, s, self.W, "bfloat16"))

    def test_train_requires_forward_to_fit(self):
        r = doc.train_precheck(L["L5"], {"L5": {"infer": {"status": "FAIL_OOM"}}})
        self.assertEqual((r["status"], r["evidence"]), ("SKIPPED_FLOOR", "floor"))
        r = doc.train_precheck(L["L5"], {"L5": {"infer": {"status": "SKIPPED_FLOOR"}}})
        self.assertEqual(r["status"], "SKIPPED_FLOOR")

    def test_train_runs_after_slow_or_marginal_inference(self):
        self.assertIsNone(doc.train_precheck(L["L3"], {"L3": {"infer": {"status": "TOO_SLOW", "params": 450e6}}}))
        self.assertIsNone(doc.train_precheck(L["L3"], {"L3": {"infer": {"status": "MARGINAL", "params": 450e6}}}))

    def test_train_inherits_not_run(self):
        r = doc.train_precheck(L["L4"], {"L4": {"infer": {"status": "FAIL_DOWNLOAD", "zh": "x", "en": "y"}}})
        self.assertEqual((r["status"], r["evidence"]), ("NOT_RUN", "not_run"))

    def test_train_oom_is_monotonic(self):
        results = {"L4": {"infer": {"status": "PASS", "params": 3.0e9}, "train": {"status": "FAIL_OOM", "params": 3.0e9}},
                   "L5": {"infer": {"status": "PASS", "params": 3.6e9}}}
        r = doc.train_precheck(L["L5"], results)
        self.assertEqual(r["status"], "SKIPPED_FLOOR")


class ClassifyExitTests(unittest.TestCase):
    def test_oom(self):
        self.assertEqual(doc.classify_exit(1, "torch.OutOfMemoryError: CUDA out of memory"), "FAIL_OOM")

    def test_sigkill_is_ram(self):
        self.assertEqual(doc.classify_exit(-9, "Loading weights"), "FAIL_RAM")
        self.assertEqual(doc.classify_exit(3221225477, ""), "FAIL_RAM")

    def test_missing_extra(self):
        self.assertEqual(doc.classify_exit(1, "ImportError: 'transformers' is required but not installed. Install it with: pip install 'lerobot[pi]'"), "FAIL_DEP")

    def test_gated(self):
        self.assertEqual(doc.classify_exit(1, "GatedRepoError: 403 Client Error"), "BLOCKED_GATED")

    def test_network(self):
        self.assertEqual(doc.classify_exit(1, "requests.exceptions.ConnectionError: Max retries exceeded"), "FAIL_DOWNLOAD")

    def test_xet_download_error_is_network(self):
        self.assertEqual(doc.classify_exit(1, "RuntimeError: Task error: File reconstruction error: CAS Client Error: Format error: I/O error: error decoding response body"), "FAIL_DOWNLOAD")

    def test_unknown_is_crash(self):
        self.assertEqual(doc.classify_exit(1, "KeyError: 'observation.images.up'"), "FAIL_CRASH")


class RealtimeAndProjectionTests(unittest.TestCase):
    def test_realtime_bands(self):
        self.assertEqual(doc.realtime_status(0.18, 50)[0], "PASS")      # budget 1.67 s
        self.assertEqual(doc.realtime_status(1.2, 50)[0], "MARGINAL")
        self.assertEqual(doc.realtime_status(2.0, 50)[0], "TOO_SLOW")

    def test_extra_training_steps_fit_the_budget(self):
        self.assertEqual(doc.extra_training_steps(0.12), 300)     # GPU: full 300
        self.assertEqual(doc.extra_training_steps(3.0), 100)      # CPU: 300 s / 3 s
        self.assertEqual(doc.extra_training_steps(900.0), 1)

    def test_projection_formula(self):
        # 5 epochs x 45000 frames / batch 8 = 28125 steps x 0.2 s = 1.5625 h
        self.assertAlmostEqual(doc.projected_hours(0.2, 8), 1.5625, places=4)


class VerdictTests(unittest.TestCase):
    def test_pass_verdict_mentions_numbers(self):
        v = doc.infer_verdict(L["L3"], {"status": "PASS", "latency_ms": 182.0, "budget_ms": 1667.0, "demo": {"status": "PASS", "hz": 29.7}})
        self.assertEqual((v["mark"], v["evidence"]), ("ok", "measured"))
        self.assertIn("182", v["zh"])
        self.assertIn("30 Hz", v["en"])   # 29.7 rounds to 30

    def test_bad_needs_measured_or_floor(self):
        for st in ("TOO_SLOW", "FAIL_OOM", "FAIL_RAM"):
            v = doc.infer_verdict(L["L1"], {"status": st, "latency_ms": 1.0, "budget_ms": 1.0})
            self.assertEqual((v["mark"], v["evidence"]), ("bad", "measured"), st)
        v = doc.infer_verdict(L["L5"], {"status": "SKIPPED_FLOOR", "zh": "权重 7.2 GB > 4 GB", "en": "w"})
        self.assertEqual((v["mark"], v["evidence"]), ("bad", "floor"))
        self.assertIn("7.2 GB", v["zh"])

    def test_not_run_is_never_bad(self):
        for st in ("BLOCKED_GATED", "FAIL_DEP", "FAIL_DOWNLOAD", "TIMEOUT", "FAIL_CRASH", "NOT_RUN"):
            v = doc.infer_verdict(L["L4"], {"status": st})
            self.assertEqual((v["mark"], v["evidence"]), ("skip", "not_run"), st)

    def test_train_bands(self):
        ok = doc.train_verdict(L["L1"], {"status": "PASS", "hours": 0.8, "batch": 8, "update_s": 0.1})
        night = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 9.5, "batch": 4, "update_s": 0.6})
        slow = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 30.0, "batch": 4, "update_s": 2.0})
        one = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 1.0, "batch": 1, "update_s": 0.1})
        self.assertEqual((ok["mark"], ok["cloud"]), ("ok", False))
        self.assertEqual((night["mark"], night["cloud"]), ("warn", False))
        self.assertEqual((slow["mark"], slow["cloud"]), ("warn", True))
        self.assertEqual(one["cloud"], True)
        self.assertIn("batch", one["zh"])
        oom = doc.train_verdict(L["L5"], {"status": "FAIL_OOM"})
        self.assertEqual((oom["mark"], oom["cloud"], oom["evidence"]), ("bad", True, "measured"))


def V(infer_mark, infer_ev="measured", train_mark="skip", cloud=None):
    return {"infer": {"mark": infer_mark, "evidence": infer_ev, "zh": "z", "en": "e"},
            "train": {"mark": train_mark, "evidence": "measured", "cloud": cloud, "zh": "z", "en": "e"}}

    def test_crashed_probe_names_error_and_log(self):
        r = {"status": "FAIL_CRASH", "error": "KeyError: 'observation.images.up'", "log": "/x/logs/L3-infer.log"}
        v = doc.infer_verdict(L["L3"], r)
        self.assertEqual(v["mark"], "skip")
        for text in (v["zh"], v["en"]):
            self.assertIn("KeyError", text)
            self.assertIn("/x/logs/L3-infer.log", text)


class RouteTests(unittest.TestCase):
    def test_r1_picks_highest_fully_local_level(self):
        lv = {"L1": V("ok", train_mark="ok", cloud=False), "L2": V("ok", train_mark="ok", cloud=False),
              "L3": V("ok", train_mark="warn", cloud=False), "L4": V("ok", train_mark="bad", cloud=True),
              "L5": V("ok", train_mark="bad", cloud=True)}
        r = doc.route_verdict(lv)
        self.assertEqual((r["rule"], r["level"]), ("R1", "L3"))
        self.assertIn("SmolVLA", r["zh"])

    def test_r2_when_only_cloud_training(self):
        lv = {"L3": V("ok", train_mark="bad", cloud=True), "L5": V("warn", train_mark="bad", cloud=True)}
        r = doc.route_verdict(lv)
        self.assertEqual((r["rule"], r["level"]), ("R2", "L5"))

    def test_r3_when_training_unknown(self):
        lv = {"L1": V("ok", train_mark="skip", cloud=None)}
        self.assertEqual(doc.route_verdict(lv)["rule"], "R3")

    def test_r4_all_measured_bad(self):
        lv = {"L1": V("bad"), "L3": V("bad", "floor")}
        self.assertEqual(doc.route_verdict(lv)["rule"], "R4")

    def test_r5_nothing_tested(self):
        lv = {"L1": V("skip", "not_run"), "L3": V("skip", "not_run")}
        r = doc.route_verdict(lv)
        self.assertEqual(r["rule"], "R5")

    def test_bad_infer_with_not_run_neighbour_is_r5_not_r4(self):
        lv = {"L1": V("bad"), "L3": V("skip", "not_run")}
        self.assertEqual(doc.route_verdict(lv)["rule"], "R5")


class EvaluateTests(unittest.TestCase):
    def test_install_failure_short_circuits(self):
        rep = {"specs": specs(), "install": {"status": "FAIL"}}
        v = doc.evaluate(rep)
        self.assertEqual(v["route"]["rule"], "R0")
        self.assertEqual(v["levels"], {})

    def test_full_report_gets_every_level(self):
        rep = {"specs": specs(nvidia=nvidia()), "install": {"status": "PASS", "torchcodec": "0.11"},
               "dataset": {"status": "PASS"},
               "levels": {"L1": {"infer": {"status": "PASS", "latency_ms": 20, "budget_ms": 3333},
                                 "train": {"status": "PASS", "hours": 0.5, "batch": 8, "update_s": 0.05}}}}
        v = doc.evaluate(rep)
        self.assertEqual(set(v["levels"]), {"L1", "L2", "L3", "L4", "L5"})
        self.assertEqual(v["route"]["rule"], "R1")
        self.assertEqual(v["levels"]["L5"]["infer"]["mark"], "skip")


class ConsoleTests(unittest.TestCase):
    def test_ascii_fallback_when_stream_cannot_encode(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        con = doc.Console(stream=stream)
        self.assertTrue(con.ascii)
        self.assertEqual(con.mark("ok"), "[OK]")

    def test_utf8_stream_uses_emoji(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        self.assertFalse(doc.Console(stream=stream).ascii)

    def test_pad_counts_cjk_as_two_cells(self):
        self.assertEqual(doc.dwidth("内存 RAM"), 8)
        self.assertEqual(len(doc.pad("ab", 5)), 5)


class ProgressTests(unittest.TestCase):
    def test_log_mode_prints_each_quarter_once(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        con = doc.Console(stream=stream)          # a BytesIO wrapper is not a tty -> log mode
        for i in range(1, 301, 15):
            con.bar("demo", i, 300)
        con.bar("demo", 300, 300)
        stream.seek(0)
        lines = stream.read().splitlines()
        self.assertEqual([l.split()[-2] for l in lines], ["1/300", "76/300", "151/300", "226/300", "300/300"])

    def test_status_line_is_cut_to_the_terminal_width(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        con = doc.Console(stream=stream)
        con.tty = True
        wide = "加载模型 loading the model " * 20
        fitted = con._fit(wide)
        self.assertLess(doc.dwidth(fitted), doc.shutil.get_terminal_size((100, 24)).columns)
        self.assertTrue(fitted.endswith("…"))


class ReleaseConsistencyTests(unittest.TestCase):
    """The README one-liners fetch the launcher from `main` (a URL that never changes); the launcher
    fetches the program at the tag baked into it, which must be the version in this file."""

    def test_launcher_default_tag_is_the_tool_version(self):
        tag = f"v{doc.TOOL_VERSION}"
        sh = (ROOT / "doctor.sh").read_text(encoding="utf-8")
        ps = (ROOT / "doctor.ps1").read_text(encoding="utf-8")
        self.assertIn(f'DOCTOR_TAG="${{DOCTOR_TAG:-{tag}}}"', sh)
        self.assertIn(f'else {{ "{tag}" }}', ps)

    def test_readme_one_liners_point_at_main(self):
        import re
        main = "https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/main/"
        for readme in (ROOT / "README.md", ROOT / "docs" / "i18n" / "zh" / "README.md"):
            text = readme.read_text(encoding="utf-8")
            self.assertIn(main + "doctor.sh | bash", text, readme)
            self.assertIn(main + "doctor.ps1 | iex", text, readme)
            self.assertEqual(re.findall(r"lerobot-doctor/v\d+\.\d+\.\d+/", text), [], readme)

    def test_ps1_is_pure_ascii(self):
        """Windows PowerShell 5.1 reads a BOM-less .ps1 in the ANSI code page and a BOM breaks `irm | iex`."""
        raw = (ROOT / "doctor.ps1").read_bytes()
        bad = [i for i, b in enumerate(raw) if b >= 0x80]
        self.assertEqual(bad, [], f"non-ASCII byte at offset {bad[:1]}")

    def test_ps1_exits_only_in_file_mode(self):
        """Under `irm | iex` a top-level `exit` closes the user's own window; the only exit sits
        inside the `if ($PSScriptRoot)` block, which is true only when started as a file."""
        import re
        lines = (ROOT / "doctor.ps1").read_text(encoding="ascii").splitlines()
        exits = [i for i, l in enumerate(lines) if re.match(r"\s*exit\b", l)]
        self.assertEqual(len(exits), 1, exits)
        start = next(i for i, l in enumerate(lines) if l.startswith("if ($PSScriptRoot) {"))
        self.assertGreater(exits[0], start)
        self.assertTrue(all(not l.startswith("}") for l in lines[start + 1:exits[0]]), "exit is outside the block")

    def test_ps1_chinese_escapes_match_the_readable_table(self):
        import codecs
        import re
        src = (ROOT / "doctor.ps1").read_text(encoding="ascii")
        literals = re.findall(r"'((?:[^']*\\u[0-9a-fA-F]{4}[^']*)+)'", src)
        decoded = {codecs.decode(lit, "unicode_escape") for lit in literals}
        self.assertEqual(decoded, set(PS1_ZH.values()))

    def test_every_launcher_announces_itself(self):
        """DOCTOR_LAUNCHER tells the program that something outside it will hold the window open."""
        self.assertIn('$env:DOCTOR_LAUNCHER = "ps1"', (ROOT / "doctor.ps1").read_text(encoding="ascii"))
        self.assertIn('set "DOCTOR_LAUNCHER=bat"', (ROOT / "doctor.bat").read_text(encoding="utf-8"))
        self.assertIn("export DOCTOR_LAUNCHER=sh", (ROOT / "doctor.sh").read_text(encoding="utf-8"))


# Readable form of every \u-escaped string in doctor.ps1 (the file itself must stay pure ASCII).
PS1_ZH = {
    "downloading": "下载体检程序",
    "installing uv": "安装 uv（Python 环境管理器，约 30 MB）",
    "preparing Python": "准备 Python 3.12",
    "uv could not start (before the code)": "uv 没能把体检程序跑起来，看上面的输出（退出码 ",
    "uv could not start (after the code)": "）",
    "launcher failed": "启动器出错了，体检没有开始",
    "step": "卡在",
    "error": "错误",
    "network hint": "最常见是网络问题：重试一次；下载慢可以把 HF_ENDPOINT 设成镜像",
    "report it": "报 issue 请把这个窗口截图贴上去",
    "press Enter": "按回车关闭窗口",
}


class CrashHandlerTests(unittest.TestCase):
    """An unexpected exception must end in a crash log and a message that names it, never a bare exit."""

    def _run_main_raising(self, exc):
        import contextlib
        import os
        import tempfile
        from unittest.mock import patch
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(doc, "WORK_DIR", Path(tmp)), \
                patch.object(doc, "_main", side_effect=exc), \
                patch.dict(os.environ, {"DOCTOR_LAUNCHER": "test"}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = doc.main(["--specs-only", "--ascii"])
            logs = sorted((Path(tmp) / "logs").glob("crash-*.log"))
            texts = [p.read_text(encoding="utf-8") for p in logs]
        return rc, out.getvalue(), err.getvalue(), logs, texts

    def test_unexpected_exception_writes_crash_log_and_says_where(self):
        rc, out, err, logs, texts = self._run_main_raising(RuntimeError("boom"))
        self.assertEqual(rc, doc.EXIT_CRASH)
        self.assertEqual(len(logs), 1)
        for needle in ("LeRobot Doctor", "role: launcher", "platform:", "python:", "Traceback", "RuntimeError: boom"):
            self.assertIn(needle, texts[0])
        self.assertIn(str(logs[0]), out)              # the path is on screen
        self.assertIn("RuntimeError: boom", out)      # and so is the reason
        self.assertIn(doc.ISSUES_URL, out)
        self.assertIn("Traceback", err)               # the full traceback stays on stderr

    def test_keyboard_interrupt_still_returns_130(self):
        rc, out, err, logs, _ = self._run_main_raising(KeyboardInterrupt())
        self.assertEqual(rc, 130)
        self.assertEqual(logs, [])

    def test_crash_log_falls_back_to_the_temp_dir(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.NamedTemporaryFile() as f:            # a file: mkdir under it fails on every OS
            try:
                raise ValueError("no home")
            except ValueError as e:
                with patch.object(doc, "WORK_DIR", Path(f.name) / "lerobot-doctor"):
                    path = doc.write_crash_log(e, "test", None)
        try:
            self.assertEqual(path.parent, Path(tempfile.gettempdir()))
            self.assertTrue(path.name.startswith("lerobot-doctor-crash-"))
            self.assertIn("ValueError: no home", path.read_text(encoding="utf-8"))
        finally:
            path.unlink()


class ReportRenderTests(unittest.TestCase):
    """The final report is two complete boxes from the same data, English first, then Chinese."""

    def setUp(self):
        import json
        self.report = json.loads((ROOT / "examples" / "linux-ubuntu24-rtx5070ti-16gb.json").read_text(encoding="utf-8"))
        self.verdicts = doc.evaluate(self.report)

    def _lines(self, draw):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        draw(doc.Console(stream=stream))
        stream.seek(0)
        return [l for l in stream.read().splitlines() if l.strip()]

    def test_english_box_has_no_chinese_and_even_borders(self):
        import unicodedata
        box = self._lines(lambda con: doc.render_one(con, self.report, self.verdicts, "en", 100))
        self.assertTrue(box[0].startswith("╔"))
        self.assertEqual({doc.vis_width(l) for l in box}, {100})
        self.assertEqual([c for l in box for c in l if unicodedata.east_asian_width(c) in ("W", "F")], [])
        self.assertIn("· Report", box[1])

    def test_chinese_box_is_chinese_and_even_borders(self):
        box = self._lines(lambda con: doc.render_one(con, self.report, self.verdicts, "zh", 100))
        self.assertEqual({doc.vis_width(l) for l in box}, {100})
        self.assertIn("· 体检报告", box[1])
        self.assertNotIn("Inference", "".join(box))

    def test_full_report_is_english_then_chinese_then_the_path(self):
        lines = self._lines(lambda con: doc.render_report(con, self.report, self.verdicts, Path("/x/report.json")))
        text = "\n".join(lines)
        self.assertLess(text.index("· Report"), text.index("· 体检报告"))
        self.assertEqual(text.count("LeRobot Doctor"), 2)
        self.assertIn("/x/report.json", lines[-1])

class WorkerProtocolTests(unittest.TestCase):
    def test_result_line_is_json(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doc.emit_result({"status": "PASS", "x": 1})
            doc.emit("progress", "timing", 3, 20, "median 10 ms")
            doc.emit("joints", 7, *["1.0"] * 6, "29.5", "12.0")
        lines = buf.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("@@result {"))
        self.assertEqual(lines[1], "@@progress timing 3 20 median 10 ms")
        self.assertEqual(lines[2].split()[1], "7")


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
