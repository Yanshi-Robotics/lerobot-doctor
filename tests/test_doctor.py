"""Unit tests for the pure parts of lerobot_doctor.py: spec parsers, accelerator classification,
hard floors, the ladder state machine, verdicts and the route rules. No network, no torch.

    python tests/test_doctor.py
"""

import contextlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_missing_params_never_crash_the_prechecks(self):
        """A worker that OOMs before counting its parameters leaves params=None; `None <= 0` was a TypeError."""
        results = {"L1": {"infer": {"status": "PASS", "params": None}, "train": {"status": "FAIL_OOM", "params": None}},
                   "L2": {"infer": {"status": "PASS", "params": None}}}
        self.assertEqual(doc.train_precheck(L["L2"], results)["status"], "SKIPPED_FLOOR")
        s = specs(nvidia=nvidia())
        results = {"L3": {"infer": {"status": "FAIL_OOM", "dtype": "bfloat16"}}}
        self.assertEqual(doc.infer_precheck(L["L4"], results, s, {"L3": {"params": None}, "L4": {"params": 1e9}}, "bfloat16")["status"], "SKIPPED_FLOOR")

    def test_cpu_weight_floor_names_the_ram_once(self):
        """On a CPU the 'device memory' is the RAM: the floor must not say 'device memory 16 GB'."""
        s = specs(ram_gb=16.0)
        r = doc.infer_precheck(L["L5"], {}, s, {"L5": {"params": 4.2e9}}, "float32")
        self.assertEqual(r["status"], "SKIPPED_FLOOR")
        self.assertIn("RAM", r["en"])
        self.assertNotIn("device memory", r["en"])

    # ---- speed monotonic: a smaller level that already needs the cloud on this CPU settles the larger ones ----
    def test_cpu_slower_level_needing_cloud_skips_training(self):
        results = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "PASS", "hours": 23.2, "batch": 8, "device": "cpu"}},
                   "L2": {"infer": {"status": "TOO_SLOW", "params": 2.6e8}}}
        r = doc.train_precheck(L["L2"], results)
        self.assertEqual((r["status"], r["evidence"], r["reason"]), ("SKIPPED_SLOWER", "floor", "slower_level_cloud:L1"))
        self.assertIn("L1", r["en"])
        self.assertIn("ACT", r["zh"])

    def test_speed_rule_is_cpu_only(self):
        results = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "PASS", "hours": 23.2, "batch": 8, "device": "cuda"}},
                   "L2": {"infer": {"status": "PASS", "params": 2.6e8}}}
        self.assertIsNone(doc.train_precheck(L["L2"], results))

    def test_speed_rule_accepts_partial_hours(self):
        results = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "TIMEOUT", "hours": 536.7, "batch": 8, "device": "cpu"}},
                   "L2": {"infer": {"status": "PASS", "params": 2.6e8}}}
        self.assertEqual(doc.train_precheck(L["L2"], results)["status"], "SKIPPED_SLOWER")

    def test_speed_rule_batch_one_counts_and_short_local_runs_do_not(self):
        base = {"L2": {"infer": {"status": "PASS", "params": 2.6e8}}}
        one = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "PASS", "hours": 3.0, "batch": 1, "device": "cpu"}}, **base}
        fast = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "PASS", "hours": 3.0, "batch": 8, "device": "cpu"}}, **base}
        self.assertEqual(doc.train_precheck(L["L2"], one)["status"], "SKIPPED_SLOWER")
        self.assertIsNone(doc.train_precheck(L["L2"], fast))

    def test_memory_rule_wins_over_speed(self):
        results = {"L1": {"infer": {"status": "PASS"}, "train": {"status": "PASS", "hours": 30.0, "batch": 8, "device": "cpu"}},
                   "L2": {"infer": {"status": "FAIL_RAM"}}}
        self.assertEqual(doc.train_precheck(L["L2"], results)["status"], "SKIPPED_FLOOR")

    def test_status_sets(self):
        self.assertIn("SKIPPED_SLOWER", doc.STATUS_FLOOR)
        self.assertNotIn("SKIPPED_SLOWER", doc.MEMORY_FAILS)


class ClassifyExitTests(unittest.TestCase):
    def test_oom(self):
        self.assertEqual(doc.classify_exit(1, "torch.OutOfMemoryError: CUDA out of memory"), "FAIL_OOM")

    def test_sigkill_is_ram(self):
        self.assertEqual(doc.classify_exit(-9, "Loading weights"), "FAIL_RAM")
        self.assertEqual(doc.classify_exit(137, ""), "FAIL_RAM")

    def test_windows_access_violation_is_the_tools_problem_not_memory(self):
        """0xC0000005 is a DLL / driver / CPU-flag failure of the stack. Calling it 'out of RAM' with
        evidence=measured made every larger level skip and the route say R4."""
        self.assertEqual(doc.classify_exit(3221225477, ""), "FAIL_CRASH")
        self.assertEqual(doc.classify_exit(-1073741819, "Loading weights"), "FAIL_CRASH")

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


class TimingTests(unittest.TestCase):
    """The realtime band is decided on p95: the arm waits for the slowest chunk, not the typical one."""

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(doc.percentile([1, 2, 3, 4, 5], 0.95), 4.8)   # int(0.95 * 4) = 3 -> 4 was the 80th
        self.assertEqual(doc.percentile([7], 0.95), 7)
        self.assertEqual(doc.percentile([3, 1, 2], 0.5), 2)
        with self.assertRaises(ValueError):
            doc.percentile([], 0.5)

    def test_infer_summary_decides_on_p95(self):
        s = doc.infer_summary([0.5] * 4 + [1.2], 32)   # budget 1.067 s; median 0.5 s would say PASS
        self.assertEqual(s["status"], "MARGINAL")
        self.assertEqual(s["latency_ms"], 500.0)
        self.assertEqual(s["p95_ms"], 1060.0)
        self.assertEqual((s["budget_ms"], s["timed_calls"]), (1067, 5))

    def test_examples_keep_their_bands_on_p95(self):
        """Switching the rule from median to p95 flips no stored verdict on either real machine."""
        for name in ("linux-ubuntu24-cpu-only-9800x3d.json", "linux-ubuntu24-rtx5070ti-16gb.json"):
            rep = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))
            for lid, lv in rep["levels"].items():
                inf = lv.get("infer") or {}
                if inf.get("p95_ms") and inf.get("n_action_steps"):
                    self.assertEqual(doc.realtime_status(inf["p95_ms"] / 1000, inf["n_action_steps"])[0], inf["status"], f"{name} {lid}")

    def test_small_sample_is_flagged(self):
        r = {"status": "TOO_SLOW", "latency_ms": 40000.0, "p95_ms": 40000.0, "budget_ms": 3333, "timed_calls": 1}
        v = doc.infer_verdict(L["L1"], r)
        self.assertIn("(n=1)", v["en"])
        self.assertIn("n=1", v["zh"])
        self.assertIn("n=1", doc.short_infer(v, r, doc.REPORT_TEXT["en"]))
        full = {**r, "timed_calls": 5}
        self.assertNotIn("n=", doc.infer_verdict(L["L1"], full)["en"])
        self.assertNotIn("n=", doc.short_infer(v, full, doc.REPORT_TEXT["en"]))

    def test_verdict_shows_p95_when_it_differs_from_the_median(self):
        v = doc.infer_verdict(L["L1"], {"status": "MARGINAL", "latency_ms": 3060.0, "p95_ms": 3390.0, "budget_ms": 3333})
        self.assertIn("p95 3390", v["en"])
        self.assertIn("p95 3390", v["zh"])


class DemoTests(unittest.TestCase):
    """The simulated task measures the policy: PASS means it kept up with the 30 Hz arm."""

    def test_demo_summary_bands(self):
        ok = doc.demo_summary(300, 300, 10.0, [0.05])
        self.assertEqual((ok["status"], ok["hz"], ok["refill_ms"], ok["target_hz"]), ("PASS", 30.0, 50.0, 30))
        slow = doc.demo_summary(300, 300, 16.5, [])
        self.assertEqual((slow["status"], slow["hz"], slow["refill_ms"]), ("SLOW", 18.2, None))
        cut = doc.demo_summary(120, 300, 61.0, [0.5, 0.7])
        self.assertEqual((cut["status"], cut["steps"], cut["refill_ms"]), ("TOO_SLOW", 120, 600.0))
        self.assertEqual(doc.demo_summary(0, 300, 0.0, [])["hz"], 0.0)

    def test_demo_text_prints_measured_numbers(self):
        zh, en = doc.demo_text({"status": "SLOW", "hz": 18.2, "seconds": 16.5, "steps": 300, "target_hz": 30, "refill_ms": 3060.0})
        self.assertIn("16.5 s", en)
        self.assertIn("16.5 s", zh)
        self.assertIn("18 Hz", en)
        self.assertIn("3060 ms", en)
        zh, en = doc.demo_text({"status": "PASS", "hz": 29.7, "seconds": 10.3})
        self.assertIn("10.3 s", en)
        self.assertNotIn("10 s", en)
        zh, en = doc.demo_text({"status": "TOO_SLOW", "hz": 2.5, "seconds": 60.2, "steps": 150})
        self.assertIn("150 steps", en)
        self.assertEqual(doc.demo_text(None), ("", ""))
        self.assertEqual(doc.demo_text({}), ("", ""))
        self.assertEqual(doc.demo_text({"status": "SKIPPED"}), ("", ""))

    def test_marginal_verdict_carries_a_slow_demo(self):
        r = {"status": "MARGINAL", "latency_ms": 666.0, "p95_ms": 679.0, "budget_ms": 1067,
             "demo": {"status": "SLOW", "hz": 18.2, "seconds": 16.5, "steps": 300, "target_hz": 30}}
        v = doc.infer_verdict(L["L2"], r)
        self.assertEqual(v["mark"], "warn")
        self.assertIn("only 18 Hz", v["en"])
        self.assertIn("18 Hz slow", doc.short_infer(v, r, doc.REPORT_TEXT["en"]))
        self.assertIn("18 Hz 偏慢", doc.short_infer(v, r, doc.REPORT_TEXT["zh"]))

    def test_old_report_shape_still_renders(self):
        """0.1.0 JSON: a demo dict with status PASS and hz only."""
        v = doc.infer_verdict(L["L3"], {"status": "PASS", "latency_ms": 182.0, "budget_ms": 1667.0, "demo": {"status": "PASS", "hz": 29.7}})
        self.assertIn("30 Hz", v["en"])


class VerdictTests(unittest.TestCase):
    def test_pass_verdict_mentions_numbers(self):
        v = doc.infer_verdict(L["L3"], {"status": "PASS", "latency_ms": 182.0, "budget_ms": 1667.0, "demo": {"status": "PASS", "hz": 29.7, "seconds": 10.1}})
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

    def test_every_status_renders_in_both_verdicts(self):
        """Table-driven: no status, and no missing number, may raise inside a verdict."""
        statuses = doc.STATUS_MEASURED_OK | doc.STATUS_MEASURED_FAIL | doc.STATUS_FLOOR | doc.STATUS_NOT_RUN
        for st in statuses:
            for r in ({"status": st}, {"status": st, "latency_ms": None, "budget_ms": None, "demo": None},
                      {"status": st, "hours": 1.0, "batch": 8, "update_s": 0.1, "partial": {}}):
                for fn, cell in ((doc.infer_verdict, doc.short_infer), (doc.train_verdict, doc.short_train)):
                    v = fn(L["L3"], r)
                    self.assertIn(v["mark"], ("ok", "warn", "bad", "skip"), (st, fn.__name__))
                    self.assertIn(v["evidence"], ("measured", "floor", "not_run", "partial"), (st, fn.__name__))
                    self.assertTrue(v["zh"] and v["en"], (st, fn.__name__))
                    for lang in ("en", "zh"):   # the table cell must render from the same record
                        self.assertTrue(cell(v, r, doc.REPORT_TEXT[lang]), (st, cell.__name__))

    def test_train_bands(self):
        ok = doc.train_verdict(L["L1"], {"status": "PASS", "hours": 0.8, "batch": 8, "update_s": 0.1})
        night = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 9.5, "batch": 4, "update_s": 0.6})
        slow = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 30.0, "batch": 4, "update_s": 2.0})
        one = doc.train_verdict(L["L3"], {"status": "PASS", "hours": 1.0, "batch": 1, "update_s": 0.1})
        self.assertEqual((ok["mark"], ok["cloud"]), ("ok", False))
        self.assertEqual((night["mark"], night["cloud"]), ("warn", False))
        self.assertEqual((slow["mark"], slow["cloud"]), ("warn", True))
        self.assertEqual((one["mark"], one["cloud"]), ("warn", True))   # a cloud verdict is never a green cell
        self.assertIn("batch", one["zh"])
        oom = doc.train_verdict(L["L5"], {"status": "FAIL_OOM"})
        self.assertEqual((oom["mark"], oom["cloud"], oom["evidence"]), ("bad", True, "measured"))

    def test_train_timeout_with_steps_is_a_partial_projection(self):
        r = {"status": "TIMEOUT", "seconds": 900.2, "partial": {"phase": "x", "step_times": [67.3, 70.1], "batch": 8}}
        v = doc.train_verdict(L["L1"], r)
        self.assertEqual((v["mark"], v["cloud"], v["evidence"]), ("warn", True, "partial"))
        self.assertIn("from 2 steps", v["en"])
        self.assertIn("2 步", v["zh"])
        self.assertIn("536.", v["en"])   # median 68.7 s x 28125 steps / 3600
        self.assertNotIn("disk", v["en"])

    def test_train_timeout_without_steps_names_the_phase(self):
        r = {"status": "TIMEOUT", "seconds": 900.0, "partial": {"phase": doc.bi("ACT · 加载模型", "ACT · loading model"), "step_times": [], "batch": None}}
        v = doc.train_verdict(L["L1"], r)
        self.assertEqual((v["mark"], v["evidence"], v["cloud"]), ("skip", "not_run", None))
        self.assertIn("15 min", v["en"])
        self.assertIn("last seen: ACT · loading model", v["en"])
        self.assertIn("最后在做：ACT · 加载模型", v["zh"])
        self.assertNotIn("disk", v["en"])

    def test_infer_timeout_names_the_phase_not_the_disk(self):
        r = {"status": "TIMEOUT", "seconds": 720.0, "partial": {"phase": doc.bi("预热", "warm-up"), "progress": {}, "step_times": [], "batch": None}}
        v = doc.infer_verdict(L["L2"], r)
        self.assertIn("last seen: warm-up", v["en"])
        self.assertNotIn("disk", v["en"])
        self.assertNotIn("磁盘", v["zh"])
        old = doc.infer_verdict(L["L2"], {"status": "TIMEOUT"})   # a 0.1.5 JSON without partial or seconds
        self.assertEqual(old["mark"], "skip")
        self.assertIn("within its budget", old["en"])

    def test_skipped_slower_is_warn_cloud_floor(self):
        v = doc.train_verdict(L["L2"], {"status": "SKIPPED_SLOWER", "zh": "L1 ACT 已要上云", "en": "L1 ACT already needs the cloud"})
        self.assertEqual((v["mark"], v["cloud"], v["evidence"]), ("warn", True, "floor"))
        self.assertIn("L1 ACT", v["en"])
        self.assertEqual(doc.short_train(v, {"status": "SKIPPED_SLOWER"}, doc.REPORT_TEXT["en"]), "only slower -> cloud")

    def test_split_bi(self):
        self.assertEqual(doc.split_bi(doc.bi("甲", "a")), ("甲", "a"))
        self.assertEqual(doc.split_bi("plain"), ("plain", "plain"))

    def test_partial_train_numbers(self):
        self.assertIsNone(doc.partial_train_numbers({"status": "TIMEOUT"}))
        self.assertIsNone(doc.partial_train_numbers({"partial": {"step_times": [1.0], "batch": None}}))
        pn = doc.partial_train_numbers({"partial": {"step_times": [2.0, 3.0, 4.0], "batch": 8}})
        self.assertEqual((pn["update_s"], pn["batch"], pn["n"]), (3.0, 8, 3))
        self.assertAlmostEqual(pn["hours"], doc.projected_hours(3.0, 8), places=1)

    def test_short_train_partial_cell(self):
        r = {"status": "TIMEOUT", "hours": 536.7, "batch": 8, "update_s": 68.7}
        v = doc.train_verdict(L["L1"], {**r, "partial": {"step_times": [67.3, 70.1], "batch": 8}, "seconds": 900})
        self.assertEqual(doc.short_train(v, r, doc.REPORT_TEXT["en"]), "~536.7 h · batch 8 -> cloud")

    def test_crashed_probe_names_error_and_log(self):
        r = {"status": "FAIL_CRASH", "error": "KeyError: 'observation.images.up'", "log": "/x/logs/L3-infer.log"}
        v = doc.infer_verdict(L["L3"], r)
        self.assertEqual(v["mark"], "skip")
        for text in (v["zh"], v["en"]):
            self.assertIn("KeyError", text)
            self.assertIn("/x/logs/L3-infer.log", text)

    def test_install_failure_text_separates_uv_from_lerobot(self):
        zh, en = doc.install_failure_text({"status": "FAIL_CRASH", "reason": "uv venv exited 2", "log": "/x/install.log", "hint": "UV_PYTHON_INSTALL_MIRROR"})
        self.assertIn("uv could not build", en)
        self.assertIn("UV_PYTHON_INSTALL_MIRROR", en)
        self.assertNotIn("LeRobot", en.split(":")[0])
        zh, en = doc.install_failure_text({"status": "FAIL", "reason": "uv pip install exited 1", "log": "/x/install.log"})
        self.assertIn("did not install", en)

    def test_ladder_text(self):
        t = doc.REPORT_TEXT["en"]
        self.assertEqual(doc.ladder_text({"attempts": [{"batch": 8, "update_s": 0.1}]}, t), "")
        r = {"attempts": [{"batch": 8, "status": "FAIL_OOM"}, {"batch": 4, "update_s": 0.2}], "peak_gb": 9.8}
        self.assertEqual(doc.ladder_text(r, t), "batch ladder 8 OOM -> 4 OK · peak 9.8 GB")


def V(infer_mark, infer_ev="measured", train_mark="skip", cloud=None, train_ev="measured"):
    return {"infer": {"mark": infer_mark, "evidence": infer_ev, "zh": "z", "en": "e"},
            "train": {"mark": train_mark, "evidence": train_ev, "cloud": cloud, "zh": "z", "en": "e"}}


class RouteTests(unittest.TestCase):
    def test_partial_train_evidence_counts(self):
        self.assertEqual(doc.route_verdict({"L1": V("ok", train_mark="warn", cloud=False, train_ev="partial")})["rule"], "R1")
        self.assertEqual(doc.route_verdict({"L1": V("ok", train_mark="warn", cloud=True, train_ev="partial")})["rule"], "R2")

    def test_skipped_slower_routes_to_r2(self):
        lv = {"L1": V("warn", train_mark="warn", cloud=True), "L2": V("bad", train_mark="warn", cloud=True, train_ev="floor")}
        r = doc.route_verdict(lv)
        self.assertEqual((r["rule"], r["level"]), ("R2", "L1"))

    def test_r2_mentions_a_higher_level_whose_inference_runs(self):
        lv = {"L1": V("warn", train_mark="warn", cloud=True), "L3": V("ok", train_mark="skip", cloud=None)}
        r = doc.route_verdict(lv)
        self.assertEqual((r["rule"], r["level"]), ("R2", "L1"))
        self.assertIn("SmolVLA", r["en"])
        self.assertIn("SmolVLA", r["zh"])
        same = doc.route_verdict({"L1": V("warn", train_mark="warn", cloud=True)})
        self.assertNotIn("up to", same["en"])

    def test_all_timeout_is_r5_and_empty_is_r5(self):
        lv = {"L1": V("skip", "not_run"), "L2": V("skip", "not_run"), "L3": V("skip", "not_run")}
        self.assertEqual(doc.route_verdict(lv)["rule"], "R5")
        r = doc.route_verdict({})
        self.assertEqual(r["rule"], "R5")
        self.assertIn("no level completed", r["en"])

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

    def test_uv_failure_is_not_called_lerobot_failing(self):
        rep = {"specs": specs(), "install": {"status": "FAIL_CRASH", "reason": "uv venv exited 2", "log": "/x"}}
        self.assertIn("uv could not build", doc.evaluate(rep)["route"]["en"])

    def test_dataset_failure_names_its_cause(self):
        base = {"specs": specs(), "install": {"status": "PASS"}}
        net = doc.evaluate({**base, "dataset": {"status": "FAIL_DOWNLOAD"}})["route"]["en"]
        crash = doc.evaluate({**base, "dataset": {"status": "FAIL_CRASH", "error": "RuntimeError: no decoder"}})["route"]["en"]
        self.assertIn("network", net)
        self.assertNotIn("network", crash)
        self.assertIn("no decoder", crash)

    def test_full_report_gets_every_level(self):
        rep = {"specs": specs(nvidia=nvidia()), "install": {"status": "PASS", "torchcodec": "0.11"},
               "dataset": {"status": "PASS"},
               "levels": {"L1": {"infer": {"status": "PASS", "latency_ms": 20, "budget_ms": 3333},
                                 "train": {"status": "PASS", "hours": 0.5, "batch": 8, "update_s": 0.05}}}}
        v = doc.evaluate(rep)
        self.assertEqual(set(v["levels"]), {"L1", "L2", "L3", "L4", "L5"})
        self.assertEqual(v["route"]["rule"], "R1")
        self.assertEqual(v["levels"]["L5"]["infer"]["mark"], "skip")
        self.assertEqual(v["basics"]["mark"], "skip")   # assemble / calibrate / record are not measured here

    def test_basics_notes_carry_the_accelerator_note(self):
        s = specs(other_gpus=["Intel(R) Iris(R) Plus Graphics"])
        notes = doc.basics_notes(s, {"status": "PASS", "torchcodec": None})
        text = " ".join(en for zh, en in notes)
        self.assertIn("non-NVIDIA GPU", text)
        self.assertIn("integrated", text)
        self.assertNotIn("just slower", text)
        self.assertIn("no verdict depends on it", text)
        mism = doc.basics_notes(specs(), {"status": "PASS", "torchcodec": "1", "version_mismatch": "0.5.0"})
        self.assertIn("0.5.0", mism[-1][1])


class WorkerOutputTests(unittest.TestCase):
    """The parent's parser of the worker's @@ protocol."""

    def _out(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        return doc.WorkerOutput(doc.Console(stream=stream), "ACT"), stream

    def test_step_times_skip_the_warm_up(self):
        out, _ = self._out()
        for i in (1, 2, 3):
            out.on_line(f"@@progress batch8 {i} 13 5.00 s/step", False)
        self.assertEqual(out.step_times, [])
        out.on_line("@@progress batch8 4 13 67.30 s/step peak 3.9 GB", False)
        self.assertEqual((out.step_times, out.batch), ([67.3], 8))
        self.assertEqual(out.progress["batch8"], [4, 13, "67.30 s/step peak 3.9 GB"])

    def test_new_batch_after_oom_resets_step_times(self):
        out, _ = self._out()
        out.on_line("@@progress batch8 5 13 70.10 s/step", False)
        out.on_line("@@event oom ACT batch 8: out of memory", False)
        for i in (1, 2, 3, 4):
            out.on_line(f"@@progress batch4 {i} 13 30.00 s/step", False)
        self.assertEqual((out.step_times, out.batch), ([30.0], 4))

    def test_activity_and_result_are_remembered(self):
        out, _ = self._out()
        out.on_line("@@activity ACT · warm-up", False)
        out.on_line('@@result {"status": "PASS", "params": 5}', False)
        out.on_line('@@result {"demo": {"hz": 29.0}}', False)
        self.assertEqual(out.last_activity, "ACT · warm-up")
        self.assertEqual(out.result, {"status": "PASS", "params": 5, "demo": {"hz": 29.0}})
        self.assertEqual(out.partial()["phase"], "ACT · warm-up")

    def test_truncated_progress_line_does_not_crash(self):
        out, _ = self._out()
        for text in ("@@progress batch8", "@@progress batch8 x 13", "@@progress", "@@result {not json"):
            self.assertTrue(out.on_line(text, False))
        self.assertEqual(out.step_times, [])

    def test_notes_reach_the_screen_either_way(self):
        out, stream = self._out()
        out.on_line("@@note one forward pass already 40x over budget", False)
        out.on_line("@@event note camera rename", False)
        stream.seek(0)
        text = stream.read()
        self.assertIn("40x over budget", text)
        self.assertIn("camera rename", text)

    def test_run_worker_timeout_returns_partial(self):
        def fake(cmd, con, log_path, env=None, cwd=None, timeout=None, on_line=None):
            for line in ("@@activity " + doc.bi("ACT · 预热", "ACT · warm-up"), "@@progress batch8 4 13 67.30 s/step",
                         "@@progress batch8 5 13 70.10 s/step"):
                on_line(line, False)
            return -999
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "stream_process", side_effect=fake):
            r = doc.run_worker(Path("py"), doc.Console(stream=stream), Path(tmp) / "L1-train.log", ["train"], 900, "ACT")
        self.assertEqual((r["status"], r["evidence"]), ("TIMEOUT", "not_run"))
        self.assertEqual(r["partial"]["step_times"], [67.3, 70.1])
        self.assertEqual(r["partial"]["batch"], 8)
        self.assertEqual(r["partial"]["phase"], doc.bi("ACT · 预热", "ACT · warm-up"))

    def test_run_worker_keeps_a_finished_result_whose_bonus_demo_overran(self):
        def fake(cmd, con, log_path, env=None, cwd=None, timeout=None, on_line=None):
            on_line('@@result {"status": "PASS", "batch": 8, "update_s": 0.1}', False)
            return -999
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "stream_process", side_effect=fake):
            r = doc.run_worker(Path("py"), doc.Console(stream=stream), Path(tmp) / "L1-train.log", ["train"], 900, "ACT")
        self.assertEqual((r["status"], r["batch"], r["overran_after_result"]), ("PASS", 8, True))

    def test_run_worker_failure_keeps_the_parents_bookkeeping(self):
        """A worker that wrote evidence=measured and then crashed is still not_run, and a previous
        attempt's 'out of memory' in the same log file does not classify this run."""
        def fake(cmd, con, log_path, env=None, cwd=None, timeout=None, on_line=None):
            with log_path.open("a", encoding="utf-8") as f:
                f.write("KeyError: 'observation.images.up'\n")
            on_line('@@result {"evidence": "measured", "seconds": 1}', False)
            return 1
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "stream_process", side_effect=fake):
            log = Path(tmp) / "L1-infer.log"
            log.write_text("previous attempt: CUDA out of memory\n", encoding="utf-8")   # before this run's offset
            r = doc.run_worker(Path("py"), doc.Console(stream=stream), log, ["infer"], 900, "ACT")
        self.assertEqual((r["status"], r["evidence"], r["returncode"]), ("FAIL_CRASH", "not_run", 1))
        self.assertIn("KeyError", r["error"])
        self.assertNotEqual(r["seconds"], 1)


class StreamProcessTests(unittest.TestCase):
    """The probe runner: deadlines fire in silence, the tree dies, the tail is delivered."""

    def _run(self, code, timeout, on_line=None):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        con = doc.Console(stream=stream)
        seen = []
        recorded = {}
        original = doc.kill_tree

        def spy(proc):
            recorded["proc"] = proc
            original(proc)

        def collect(text, is_cr):
            seen.append((text, is_cr))
            return bool(on_line and on_line(text, is_cr))

        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "kill_tree", side_effect=spy):
            log = Path(tmp) / "x.log"
            t0 = time.monotonic()
            rc = doc.stream_process([sys.executable, "-c", code], con, log, timeout=timeout, on_line=collect)
            wall = time.monotonic() - t0
            log_text = log.read_text(encoding="utf-8")
        return rc, wall, seen, log_text, recorded.get("proc")

    def test_timeout_fires_during_silence(self):
        rc, wall, seen, log, proc = self._run("import time; print('x', flush=True); time.sleep(5)", 0.5)
        self.assertEqual(rc, -999)
        self.assertLess(wall, 3)
        self.assertEqual([s[0] for s in seen], ["x"])
        self.assertIn("[lerobot-doctor] TIMEOUT", log)

    def test_timeout_reaps_and_closes_the_pipe(self):
        rc, wall, seen, log, proc = self._run("import time; time.sleep(5)", 0.3)
        self.assertEqual(rc, -999)
        self.assertIsNotNone(proc.returncode)
        self.assertTrue(proc.stdout.closed)

    def test_timeout_delivers_the_unterminated_tail(self):
        code = "import sys, time; sys.stdout.write('@@progress batch8 5 13 67.30 s/step'); sys.stdout.flush(); time.sleep(5)"
        rc, wall, seen, log, proc = self._run(code, 0.5)
        self.assertEqual(seen, [("@@progress batch8 5 13 67.30 s/step", False)])

    @unittest.skipIf(platform.system() == "Windows", "process groups are POSIX")
    def test_timeout_kills_the_grandchild(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            code = (f"import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                    f"open({str(pidfile)!r}, 'w').write(str(p.pid)); print('spawned', flush=True); time.sleep(30)")
            rc, wall, seen, log, proc = self._run(code, 0.5)
            pid = int(pidfile.read_text())
        self.assertEqual(rc, -999)
        for _ in range(40):
            try:
                os.kill(pid, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, 9)
            self.fail("grandchild survived the kill")

    def test_normal_exit_keeps_cr_lf_splitting(self):
        rc, wall, seen, log, proc = self._run("import sys; sys.stdout.write('a\\rb\\nc')", 10)
        self.assertEqual(rc, 0)
        self.assertEqual(seen, [("a", True), ("b", False), ("c", False)])
        self.assertIsNone(proc)   # nothing was killed

    def test_exception_from_on_line_kills_the_child(self):
        def boom(text, is_cr):
            raise RuntimeError("page died")
        with self.assertRaises(RuntimeError):
            self._run("import time; print('x', flush=True); time.sleep(30)", 10, on_line=boom)

    def test_log_since_reads_only_this_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.log"
            p.write_text("first attempt: ConnectionError\n", encoding="utf-8")
            offset = p.stat().st_size
            with p.open("a", encoding="utf-8") as f:
                f.write("second attempt: KeyError\n")
            self.assertEqual(doc.log_since(p, offset), "second attempt: KeyError\n")
            self.assertEqual(doc.log_since(Path(tmp) / "missing.log", 0), "")

    def test_build_env_passes_the_timeouts(self):
        calls = []

        def fake_stream(cmd, con, log_path, env=None, cwd=None, timeout=None, on_line=None):
            calls.append((cmd[1], timeout))
            return 0
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "WORK_DIR", Path(tmp)), \
                patch.object(doc, "find_uv", return_value="uv"), patch.object(doc, "stream_process", side_effect=fake_stream), \
                patch.object(doc, "import_probe", return_value=(0, {"lerobot": "0.6.1"}, "")):
            report = {}
            py = doc.build_env(doc.Console(stream=stream), {"torch_backend": "cpu"}, report, None)
        self.assertEqual(calls, [("venv", doc.UV_VENV_TIMEOUT_S), ("pip", doc.UV_INSTALL_TIMEOUT_S)])
        self.assertEqual(report["install"]["status"], "PASS")
        self.assertIsNotNone(py)

    def test_import_probe_takes_the_last_json_line(self):
        code = "import json; print(json.dumps({'lerobot': '0.6.1'})); print('libomp: something after')"
        with patch.object(doc, "IMPORT_PROBE", code):
            rc, info, err = doc.import_probe(Path(sys.executable))
        self.assertEqual((rc, info), (0, {"lerobot": "0.6.1"}))
        with patch.object(doc, "IMPORT_PROBE", "print('no json at all')"):
            rc, info, err = doc.import_probe(Path(sys.executable))
        self.assertEqual((rc, info), (0, None))


class ReportFileTests(unittest.TestCase):
    def test_each_run_gets_its_own_file_and_latest_follows(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(doc, "WORK_DIR", Path(tmp)):
            p = doc.new_report_path()
            self.assertRegex(p.name, r"^report-\d{4}-\d{2}-\d{2}-\d{4}\.json$")
            rep = doc.Report(p, {"a": 1, "when": Path("/x")})   # a Path must not lose the report
            rep.data["a"] = 2
            rep.save()
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["a"], 2)
            latest = json.loads((Path(tmp) / doc.LATEST_REPORT_NAME).read_text(encoding="utf-8"))
            self.assertEqual(latest["a"], 2)
            self.assertEqual(sorted(x.name for x in Path(tmp).iterdir()), sorted([p.name, doc.LATEST_REPORT_NAME]))   # no .tmp left behind

    def test_provenance_and_version_flag(self):
        with patch.dict(os.environ, {"DOCTOR_TAG": "v9.9.9", "DOCTOR_LAUNCHER": "sh"}):
            prov = doc.provenance()
        self.assertEqual((prov["doctor_tag"], prov["launcher"]), ("v9.9.9", "sh"))
        self.assertEqual(prov["python"]["executable"], sys.executable)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            doc.main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(doc.TOOL_VERSION, out.getvalue())


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

    def test_launchers_export_the_tag(self):
        """The program records which tag ran it; both launchers hand the tag over in the environment."""
        self.assertIn("export DOCTOR_TAG", (ROOT / "doctor.sh").read_text(encoding="utf-8"))
        ps = (ROOT / "doctor.ps1").read_text(encoding="ascii")
        self.assertIn("$env:DOCTOR_TAG = $DoctorTag", ps)
        self.assertIn("Remove-Item Env:DOCTOR_TAG", ps)

    def test_line_endings_per_launcher(self):
        """doctor.bat needs CRLF (cmd misparses LF); the three Unix-side files must be LF."""
        self.assertEqual((ROOT / "doctor.bat").read_bytes().count(b"\r\n"), (ROOT / "doctor.bat").read_bytes().count(b"\n"))
        for name in ("doctor.sh", "doctor.command", "doctor.ps1"):
            self.assertNotIn(b"\r", (ROOT / name).read_bytes(), name)
        self.assertFalse((ROOT / "doctor.ps1").read_bytes().startswith(b"\xef\xbb\xbf"), "BOM breaks irm | iex")

    def test_python_is_found_before_it_is_downloaded(self):
        """A system 3.12 is enough; the GitHub download of python-build-standalone runs only without one,
        and its failure names UV_PYTHON_INSTALL_MIRROR instead of passing silently."""
        ps = (ROOT / "doctor.ps1").read_text(encoding="ascii")
        sh = (ROOT / "doctor.sh").read_text(encoding="utf-8")
        for text, name in ((ps, "ps1"), (sh, "sh")):
            self.assertLess(text.index("uv python find"), text.index("uv python install"), name)
            self.assertIn("UV_PYTHON_INSTALL_MIRROR", text, name)
            self.assertNotIn("--quiet", text, name)
        after_install = ps[ps.index("uv python install 3.12"):].splitlines()[1]
        self.assertIn("$LASTEXITCODE", after_install)
        self.assertIn("trap on_err ERR", sh)

    def test_sh_never_falls_back_to_dollar_zero(self):
        """Under `curl | bash` BASH_SOURCE is unset; falling back to $0 made a stray lerobot_doctor.py in
        the current folder run instead of the release."""
        sh = (ROOT / "doctor.sh").read_text(encoding="utf-8")
        self.assertNotIn(":-$0", sh)
        self.assertIn("BASH_SOURCE[0]:-", sh)

    def test_bat_keeps_the_exit_code_and_command_checks_for_sh(self):
        bat = (ROOT / "doctor.bat").read_text(encoding="utf-8")
        self.assertIn("%ERRORLEVEL%", bat)
        self.assertIn("exit /b %RC%", bat)
        cmd = (ROOT / "doctor.command").read_text(encoding="utf-8")
        self.assertIn("doctor.sh", cmd)
        self.assertIn("-f ./doctor.sh", cmd)

    def test_changelog_has_this_version_and_git_has_its_tag_or_the_previous(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertRegex(changelog, rf"(?m)^## \[?{re.escape(doc.TOOL_VERSION)}\]?[\s(]")
        tags = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True).stdout.split()
        if tags:   # the tag for TOOL_VERSION is created at release; until then the previous one must exist
            self.assertTrue(any(t.startswith("v0.") for t in tags))


@unittest.skipUnless(shutil.which("bash"), "bash launcher tests need bash")
class ShLauncherTests(unittest.TestCase):
    """doctor.sh driven with a fake `uv` and a fake `curl` on PATH: no network, no real install."""

    def _fake_bin(self, tmp: Path, find_rc: int, install_rc: int) -> Path:
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        uv = bin_dir / "uv"
        uv.write_text(f"""#!/usr/bin/env bash
case "$1" in
  python)
    case "$2" in
      find) if [ {find_rc} -eq 0 ]; then echo /fake/python3.12; fi; exit {find_rc} ;;
      install) echo "fake download failed" >&2; exit {install_rc} ;;
    esac ;;
  run) printf '%s\\n' "$@" >> "$UV_LOG"; exit 0 ;;
esac
exit 0
""")
        uv.chmod(0o755)
        curl = bin_dir / "curl"
        curl.write_text("""#!/usr/bin/env bash
out=""
while [ $# -gt 0 ]; do if [ "$1" = "-o" ]; then out="$2"; shift; fi; shift; done
[ -n "$out" ] && echo "# fetched" > "$out"
echo curl >> "$UV_LOG"
""")
        curl.chmod(0o755)
        return bin_dir

    def _run(self, find_rc, install_rc):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bin_dir = self._fake_bin(tmp, find_rc, install_rc)
            home = tmp / "home"
            home.mkdir()
            cwd = tmp / "cwd"
            cwd.mkdir()
            (cwd / "lerobot_doctor.py").write_text("# stray copy in the current folder\n")
            log = tmp / "uv.log"
            env = {"PATH": f"{bin_dir}:{os.environ.get('PATH', '')}", "HOME": str(home), "UV_LOG": str(log), "LANG": "C.UTF-8"}
            p = subprocess.run(["bash"], input=(ROOT / "doctor.sh").read_text(encoding="utf-8"), cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=30)
            calls = log.read_text().splitlines() if log.exists() else []
            return p.returncode, p.stdout + p.stderr, calls, home

    def test_piped_never_picks_a_stray_script_in_cwd(self):
        rc, out, calls, home = self._run(find_rc=0, install_rc=0)
        self.assertEqual(rc, 0, out)
        self.assertIn("curl", calls)
        run_args = [c for c in calls if c.endswith("lerobot_doctor.py")]
        self.assertEqual(run_args, [str(home / "lerobot-doctor" / "lerobot_doctor.py")])
        self.assertIn("/fake/python3.12", out)   # the system Python was used, nothing downloaded

    def test_python_install_failure_names_the_step_and_the_mirror(self):
        rc, out, calls, home = self._run(find_rc=2, install_rc=1)
        self.assertNotEqual(rc, 0)
        self.assertIn("install Python 3.12", out)
        self.assertIn("UV_PYTHON_INSTALL_MIRROR", out)
        self.assertNotIn("lerobot_doctor.py", "\n".join(c for c in calls if c != "curl"))

    def test_sh_parses(self):
        for name in ("doctor.sh", "doctor.command"):
            p = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)


# Readable form of every \u-escaped string in doctor.ps1 (the file itself must stay pure ASCII).
PS1_ZH = {
    "downloading": "下载体检程序",
    "installing uv": "安装 uv（Python 环境管理器，约 30 MB）",
    "preparing Python": "准备 Python 3.12",
    "python found": "已有 Python 3.12：",
    "downloading Python": "下载 Python 3.12（约 30 MB）",
    "python mirror hint": "下载 Python 3.12 失败；防火墙后请先设置镜像变量 UV_PYTHON_INSTALL_MIRROR 再重跑",
    "uv missing at run": "uv 不在 PATH 里，没法启动体检程序",
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
