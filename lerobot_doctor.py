#!/usr/bin/env python3
"""LeRobot Doctor: can this computer run LeRobot 0.6.1 for an SO-101 arm, and how far?

    python lerobot_doctor.py                # full check: specs -> install -> real inference/training
    python lerobot_doctor.py --specs-only   # only the spec table (stdlib, no download)
    python lerobot_doctor.py --uninstall    # remove ~/lerobot-doctor (keeps the Hugging Face cache)

Normal users never type these: `doctor.sh` / `doctor.ps1` install `uv`, get Python 3.12 and run
this file. The file then builds its own virtualenv, installs lerobot 0.6.1, and re-executes itself
inside that venv as the orchestrator. Every model probe runs in a child process so an out-of-memory
crash never takes the report down with it.

Everything printed is bilingual (Chinese first, English second). Every "cannot" verdict carries its
evidence: `measured` (we tried and it failed) or `floor` (an arithmetic hard floor, shown to the
user). Anything else is "not tested" with the reason.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import json
import math
import os
import platform
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

# ----------------------------------------------------------------------------------------------
# Constants. Every threshold is named and carries its source; nothing is tuned per machine.
# ----------------------------------------------------------------------------------------------

TOOL_VERSION = "0.1.6"
LEROBOT_VERSION = "0.6.1"
PYTHON_VERSION = "3.12"                         # lerobot 0.6.1: Requires-Python >=3.12
VISER_SPEC = "viser[urdf]==1.1.0"               # same pin as the Season-1 course repo
LEROBOT_EXTRAS = "smolvla,xvla,wallx,diffusion,dataset,feetech,accelerate-dep"
WORK_DIR = Path.home() / "lerobot-doctor"
LATEST_REPORT_NAME = "report-latest.json"       # a copy of the newest report; the dated files are never overwritten
DEFAULT_PORT = 4604                             # yanshirobotics 46xx range; 127.0.0.1 only
HF_OFFICIAL_ENDPOINT = "https://huggingface.co"  # the only host that gets the user's token (a mirror does not)
ISSUES_URL = "https://github.com/Yanshi-Robotics/lerobot-doctor/issues/new"
EXIT_CRASH = 70                                 # sysexits.h EX_SOFTWARE: the tool itself failed, not the machine
WINDOWS_11_FIRST_BUILD = 22000                  # platform.win32_ver() says "10" for Windows 11; the build tells them apart

MIN_FREE_DISK_GB = 30      # venv ~7.5 GB + three checkpoints ~13 GB + dataset + headroom
MIN_RAM_GB = 8             # below this torch import + any model load fails
MIN_NVIDIA_DRIVER = (570, 86)   # PyTorch cu128 wheel driver floor
TORCH_BACKEND_NVIDIA = "cu128"  # covers Ampere..Blackwell on driver >= 570.86
MIN_MACOS_FOR_MPS = (12, 3)     # PyTorch MPS backend floor
BF16_MIN_COMPUTE_CAP = 8.0      # Ampere and newer run bfloat16 natively
BYTES_PER_PARAM = {"bfloat16": 2, "float16": 2, "float32": 4}

CONTROL_FPS = 30           # the course drives the SO-101 at 30 Hz
REALTIME_RATIO = 0.5       # chunk latency <= 50 % of the chunk's play time  -> REALTIME
MARGINAL_RATIO = 1.0       # <= 100 %                                        -> MARGINAL
HOPELESS_RATIO = 12        # one CPU/MPS forward already 12x over budget: stop timing, it is TOO_SLOW
WARMUP_GPU, TIMED_GPU = 3, 20
WARMUP_CPU, TIMED_CPU = 1, 5

DEMO_SECONDS = 10          # simulated task length
DEMO_WALL_CAP_S = 60       # a slow model may take longer; we cap the wall clock
DEMO_OK_RATIO = 0.9        # the demo kept up at >= 90 % of CONTROL_FPS: a sleep-paced 30 Hz loop loses 1-2 Hz to scheduler jitter alone
DEMO_FOLLOW_ALPHA = 0.35   # first-order low-pass: how fast the simulated joints follow a target
ACT_DEMO_STEPS = 300       # extra ACT training steps so the second demo shows a learning model
ACT_DEMO_BUDGET_S = 300    # ...but never more than five minutes of it (a CPU may need seconds per step)
ACT_DEMO_MIN_STEPS = 30    # fewer extra steps than this change nothing visible; the second demo is skipped and says so

TRAIN_WARMUP, TRAIN_TIMED = 3, 10
REF_FRAMES = 45_000        # HF hardware guide: 50 episodes x 30 s x 30 fps
REF_EPOCHS = 5             # HF hardware guide: imitation learning converges in 5-10 epochs
LOCAL_OK_H = 2.0           # projected hours: comfortable locally
OVERNIGHT_H = 12.0         # projected hours: one night; above this we say "cloud"

INFER_BUDGET_S = 12 * 60
TRAIN_BUDGET_S = 15 * 60
TRAIN_L1_BUDGET_S = 25 * 60
HEARTBEAT_SILENCE_S = 2.0
NON_TTY_HEARTBEAT_S = 30.0     # log files: one "still working" line per half minute, never a stream
BAR_WIDTH = 24                 # cells in a progress bar
BAR_LOG_STEP = 25              # log files: a bar line only at 25 / 50 / 75 / 100 %
CHILD_POLL_S = 0.05            # how often the parent checks a probe's deadline, output or not
CHILD_WAIT_AFTER_KILL_S = 10    # SIGKILL / taskkill is not instant on a child that is swapping out tens of GB
UV_VENV_TIMEOUT_S = 10 * 60     # `uv venv --seed`: local work once Python 3.12 is there
UV_INSTALL_TIMEOUT_S = 60 * 60  # one `uv pip install` attempt: ~4 GB of wheels at 1 MB/s is ~70 min; a retry resumes from uv's cache
PROBE_TIMEOUT_S = 600           # importing torch + lerobot in a fresh venv, cold disk
UV_HTTP_TIMEOUT_S = 600    # uv's default 30 s drops 600 MB CUDA wheels on slow links
INSTALL_ATTEMPTS = 3       # network hiccups are the most common student failure; uv caches finished wheels
HF_HTTP_TIMEOUT_S = 60
HF_HTTP_ATTEMPTS = 2
HF_HUB_DOWNLOAD_TIMEOUT_S = 60  # huggingface_hub's default read timeout is 10 s; 13 GB of weights on a home link stall longer
HF_HUB_ETAG_TIMEOUT_S = 30

DATASET_REPO = "lerobot/svla_so101_pickplace"   # official SO-101 pick-place recording, v3.0
DATASET_EPISODES = [0, 1, 2, 3, 4]
DEMO_EPISODE = 0
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

URDF_REPO = "TheRobotStudio/SO-ARM100"
URDF_COMMIT = "7629d2ad9853d10fb903093a33ef6114099d97e5"   # same pin as the course's fetch_model.py
URDF_FILE = "Simulation/SO101/so101_new_calib.urdf"
URDF_MESHES = [
    "base_motor_holder_so101_v1.stl", "base_so101_v2.stl", "motor_holder_so101_base_v1.stl",
    "motor_holder_so101_wrist_v1.stl", "moving_jaw_so101_v1.stl", "rotation_pitch_so101_v1.stl",
    "sts3215_03a_no_horn_v1.stl", "sts3215_03a_v1.stl", "under_arm_so101_v1.stl", "upper_arm_so101_v1.stl",
    "waveshare_mounting_plate_so101_v2.stl", "wrist_roll_follower_so101_v1.stl", "wrist_roll_pitch_so101_v2.stl",
]
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"

# HF hardware guide, peak VRAM at batch 8 with AdamW, per policy group. Used ONLY for the
# pre-install estimate table; never for verdicts.
ESTIMATE_VRAM_GB = {"Light BC": (2, 6), "Diffusion": (8, 14), "Small VLA": (10, 16), "Large VLA": (24, 40)}


class Level:
    def __init__(self, lid, policy, group, source, weights, extra, batches, large=False, label=None,
                 load_dtype="float32"):
        self.id = lid
        self.policy = policy
        self.group = group
        self.source = source          # lerobot-format checkpoint loaded with from_pretrained, or None (built from config)
        self.weights = weights        # HF repo whose size/params drive downloads and the weight floor, or None (from scratch)
        self.extra = extra
        self.batches = batches
        self.large = large            # bfloat16 + gradient checkpointing where the policy config supports it
        self.load_dtype = load_dtype  # dtype lerobot materialises the checkpoint in before any cast (fp32 for all three)
        self.label = label or policy

    @property
    def pretrained(self):
        return self.weights is not None


# One representative per HF hardware-guide group, all downloadable without a Hugging Face account
# (the pi0 family needs Google's gated PaliGemma tokenizer, so it is deliberately not here).
LEVELS = [
    Level("L1", "act", "Light BC", None, None, None, [8, 4], label="ACT"),
    Level("L2", "diffusion", "Diffusion", None, None, "diffusion", [8, 4, 2], label="Diffusion"),
    Level("L3", "smolvla", "Small VLA", "lerobot/smolvla_base", "lerobot/smolvla_base", "smolvla", [8, 4, 2, 1], label="SmolVLA"),
    Level("L4", "xvla", "Large VLA", "lerobot/xvla-base", "lerobot/xvla-base", "xvla", [8, 4, 2, 1], large=True, label="X-VLA"),
    Level("L5", "wall_x", "Large VLA", None, "x-square-robot/wall-oss-flow", "wallx", [4, 2, 1], large=True, label="WALL-OSS"),
]
LEVEL_BY_ID = {lv.id: lv for lv in LEVELS}

STATUS_MEASURED_OK = {"PASS", "MARGINAL"}
STATUS_MEASURED_FAIL = {"TOO_SLOW", "FAIL_OOM", "FAIL_RAM"}
STATUS_FLOOR = {"SKIPPED_FLOOR", "SKIPPED_SLOWER"}   # SKIPPED_SLOWER is a training-only speed statement, never a memory one
STATUS_NOT_RUN = {"TIMEOUT", "FAIL_DEP", "FAIL_DOWNLOAD", "BLOCKED_GATED", "FAIL_CRASH", "NOT_RUN"}
MEMORY_FAILS = {"FAIL_OOM", "FAIL_RAM", "SKIPPED_FLOOR"}   # read to infer "the forward pass did not fit" from an inference status
BI_SEP = "  |  "


def bi(zh: str, en: str) -> str:
    """One bilingual line. Chinese first, English after a separator."""
    return f"{zh}{BI_SEP}{en}"


def split_bi(text: str) -> tuple[str, str]:
    """Undo bi(): (zh, en) from one bilingual line; a plain line serves as both."""
    zh, sep, en = text.partition(BI_SEP)
    return (zh, en) if sep else (text, text)


def percentile(values, q: float) -> float:
    """Linear interpolation between order statistics (numpy's default). `sorted(v)[int(q * (n - 1))]`
    picked the 80th percentile at n = 5."""
    s = sorted(values)
    if not s:
        raise ValueError("percentile of nothing")
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def provenance() -> dict:
    """Who ran this file: the launcher and the tag it fetched (both exported by the launchers), and
    the interpreter. A report that behaves oddly is attributed with this."""
    return {"doctor_tag": os.environ.get("DOCTOR_TAG"), "launcher": os.environ.get("DOCTOR_LAUNCHER"),
            "python": {"version": sys.version.split()[0], "executable": sys.executable}}


def gb(nbytes) -> float:
    return round(nbytes / 1e9, 1) if nbytes else 0.0


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------------------------
# Console: bilingual output, progress line, heartbeat so the screen never looks stuck.
# ----------------------------------------------------------------------------------------------

class Console:
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    SPINNER_ASCII = "-\\|/"

    def __init__(self, stream=None, ascii_only=None):
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        if ascii_only is None:
            ascii_only = not self._can_encode("✅ ⚠️ ⛔ — ⠋")
        self.ascii = ascii_only
        self.marks = (
            {"ok": "[OK]", "warn": "[!!]", "bad": "[XX]", "skip": "[--]", "run": "[..]", "info": "[  ]"}
            if ascii_only else
            {"ok": "✅", "warn": "⚠️ ", "bad": "⛔", "skip": "—", "run": "⏳", "info": "·"}
        )
        if self.tty and platform.system() == "Windows":
            os.system("")   # turns on VT escape processing in the Windows console (colours, \r)
        self._lock = threading.RLock()
        self._last_output = time.monotonic()
        self._activity = ""
        self._activity_since = time.monotonic()
        self._transient_len = 0
        self._stop = threading.Event()
        self._thread = None
        self._last_nontty_beat = 0.0
        self._bar_bucket = {}          # per bar label: last quarter written to a log file

    def _can_encode(self, text: str) -> bool:
        enc = getattr(self.stream, "encoding", None) or "ascii"
        try:
            text.encode(enc)
            return True
        except (UnicodeEncodeError, LookupError):
            return False

    def mark(self, kind: str) -> str:
        return self.marks[kind]

    def _clear_transient(self):
        if self.tty and self._transient_len:
            self.stream.write("\r" + " " * self._transient_len + "\r")
            self._transient_len = 0

    def _fit(self, text: str) -> str:
        """Cut a status line to the terminal width. A line that wraps cannot be rewound with \\r,
        and every redraw would then leave the previous row on screen (the classic scrolling spam)."""
        width = shutil.get_terminal_size((100, 24)).columns - 1
        if dwidth(text) <= width:
            return text
        out, used = [], 0
        for ch in text:
            w = dwidth(ch)
            if used + w > width - 1:
                break
            out.append(ch)
            used += w
        return "".join(out) + "…"

    def line(self, text: str = ""):
        with self._lock:
            self._clear_transient()
            self.stream.write(text + "\n")
            self.stream.flush()
            self._last_output = time.monotonic()

    def raw(self, text: str):
        """Pass-through for child/tool output that already has its own formatting."""
        with self._lock:
            self._clear_transient()
            self.stream.write(text)
            self.stream.flush()
            self._last_output = time.monotonic()

    def transient(self, text: str, log_worthy: bool = False):
        """One status line that is redrawn in place on a terminal. In a log file (no tty) it is
        printed only when `log_worthy` (a milestone) or at most once per NON_TTY_HEARTBEAT_S."""
        with self._lock:
            now = time.monotonic()
            if self.tty:
                text = self._fit(text)
                self._clear_transient()
                self.stream.write("\r" + text)
                self.stream.flush()
                self._transient_len = dwidth(text) + 1
            elif log_worthy or now - self._last_nontty_beat >= NON_TTY_HEARTBEAT_S:
                self.stream.write(text + "\n")
                self.stream.flush()
                self._last_nontty_beat = now
            self._last_output = now

    def bar(self, label: str, i: int, n: int, note: str = ""):
        """conda/npm-style bar: `label [████████░░░░░░░░] 12/20  60%  note`, redrawn in place."""
        n = max(n, 1)
        frac = min(max(i / n, 0.0), 1.0)
        filled = int(round(frac * BAR_WIDTH))
        block, empty = ("#", "-") if self.ascii else ("█", "░")
        text = f"  {label} [{block * filled}{empty * (BAR_WIDTH - filled)}] {i}/{n} {int(frac * 100):3d}%  {note}".rstrip()
        bucket = int(frac * 100) // BAR_LOG_STEP          # 0..4: which quarter we are in
        last = self._bar_bucket.get(label, -1)
        milestone = i >= n or bucket > last
        self._bar_bucket[label] = 5 if i >= n else bucket
        self.transient(text, log_worthy=milestone)

    def activity(self, text: str):
        """What the heartbeat says while nothing else is being printed."""
        with self._lock:
            self._activity = text
            self._activity_since = time.monotonic()

    def start_heartbeat(self):
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="heartbeat", daemon=True)
        self._thread.start()

    def stop_heartbeat(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        with self._lock:
            self._clear_transient()

    def _beat(self):
        i = 0
        spin = self.SPINNER_ASCII if self.ascii else self.SPINNER
        while not self._stop.wait(0.25):
            with self._lock:
                silent = time.monotonic() - self._last_output
                if not self._activity or silent < HEARTBEAT_SILENCE_S:
                    continue
                elapsed = int(time.monotonic() - self._activity_since)
                i = (i + 1) % len(spin)
                text = f"  {spin[i]} {self._activity} … {fmt_duration(elapsed)}"
            self.transient(text)
            with self._lock:
                self._last_output -= HEARTBEAT_SILENCE_S  # keep beating; transient() reset it

    # convenience -----------------------------------------------------------------------
    def step(self, k: int, total: int, zh: str, en: str, eta: str = ""):
        tail = f"（{eta}）" if eta else ""
        self.line("")
        self.line(f"[{k}/{total}] {zh}{tail}")
        self.line(f"      {en}")

    def item(self, kind: str, zh: str, en: str):
        self.line(f"  {self.mark(kind)} {bi(zh, en)}")


def dwidth(text: str) -> int:
    """Terminal display width: CJK characters take two cells."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - dwidth(text))


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ----------------------------------------------------------------------------------------------
# Stage A: specs (stdlib only, three operating systems).
# ----------------------------------------------------------------------------------------------

def run_cmd(args, timeout=20) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_nvidia_smi(text: str) -> list[dict]:
    """Rows of `--query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader`."""
    gpus = []
    for row in text.strip().splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            continue
        mem = re.search(r"([\d.]+)\s*MiB", parts[1])
        cap = None
        if len(parts) >= 4:
            try:
                cap = float(parts[3])
            except ValueError:
                cap = None
        gpus.append({
            "name": parts[0],
            "vram_gb": round(float(mem.group(1)) / 1024, 1) if mem else None,
            "driver": parts[2],
            "compute_cap": cap,
        })
    return gpus


def parse_driver_version(text: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def parse_meminfo(text: str) -> float | None:
    m = re.search(r"MemTotal:\s+(\d+)\s*kB", text)
    return round(int(m.group(1)) * 1024 / 1e9, 1) if m else None


def parse_os_release(text: str) -> str:
    m = re.search(r'^PRETTY_NAME="?(.*?)"?$', text, re.M)
    return m.group(1) if m else ""


def windows_release_name(release: str, version: str) -> str:
    """Pure. platform.win32_ver() reports Windows 11 as release "10" with version "10.0.22xxx"."""
    m = re.match(r"\d+\.\d+\.(\d+)", version or "")
    if release == "10" and m and int(m.group(1)) >= WINDOWS_11_FIRST_BUILD:
        return "11"
    return release


def find_nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if platform.system() == "Windows":
        for cand in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                     Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"):
            if cand.exists():
                return str(cand)
    return None


def collect_specs() -> dict:
    system = platform.system()
    specs = {
        "system": system,
        "os": "",
        "arch": platform.machine(),
        "kernel": platform.release(),
        "wsl": False,
        "rosetta": False,
        "cpu": "",
        "cpu_count": os.cpu_count() or 0,
        "ram_gb": None,
        "disk_free_gb": round(shutil.disk_usage(Path.home()).free / 1e9, 1),
        "python": platform.python_version(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "nvidia": [],
        "other_gpus": [],
        "gpu_query_failed": False,  # Windows: the WMI query itself failed, so "no other GPU" is unknown, not false
        "accelerator": "cpu",       # cuda | mps | cpu
        "device_mem_gb": None,      # VRAM (cuda), unified memory (mps), RAM (cpu)
        "bf16": False,
    }
    if system == "Linux":
        specs["os"] = parse_os_release(_read("/etc/os-release")) or f"Linux {platform.release()}"
        try:
            specs["glibc"] = os.confstr("CS_GNU_LIBC_VERSION")
        except (ValueError, OSError, AttributeError):
            specs["glibc"] = ""
        specs["wsl"] = "microsoft" in _read("/proc/version").lower()
        m = re.search(r"model name\s*:\s*(.+)", _read("/proc/cpuinfo"))
        specs["cpu"] = m.group(1).strip() if m else platform.processor()
        specs["ram_gb"] = parse_meminfo(_read("/proc/meminfo"))
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]")):
            vendor = _read(card / "device" / "vendor").strip()
            if vendor and vendor != "0x10de":   # 0x10de = NVIDIA, reported via nvidia-smi below
                specs["other_gpus"].append({"0x1002": "AMD", "0x8086": "Intel"}.get(vendor, vendor))
    elif system == "Darwin":
        ver = platform.mac_ver()[0]
        specs["os"] = f"macOS {ver}"
        specs["macos_version"] = tuple(int(x) for x in ver.split(".")[:2]) if ver else (0, 0)
        specs["cpu"] = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        mem = run_cmd(["sysctl", "-n", "hw.memsize"]).strip()
        specs["ram_gb"] = round(int(mem) / 1e9, 1) if mem.isdigit() else None
        specs["rosetta"] = run_cmd(["sysctl", "-n", "sysctl.proc_translated"]).strip() == "1"
        if specs["rosetta"]:
            specs["arch"] = "arm64 (Python running under Rosetta as x86_64)"
    elif system == "Windows":
        rel, ver, *_ = platform.win32_ver()
        rel = windows_release_name(rel, ver)
        specs["os"] = f"Windows {rel} ({ver})"
        specs["windows_release"] = rel
        try:
            import winreg  # type: ignore
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            specs["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:  # noqa: BLE001 - best effort on a foreign registry
            specs["cpu"] = platform.processor()
        specs["ram_gb"] = _windows_ram_gb()
        try:
            ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                                capture_output=True, text=True, timeout=20, encoding="utf-8", errors="replace")
            specs["gpu_query_failed"] = ps.returncode != 0
            if ps.returncode == 0:
                specs["other_gpus"] = [g.strip() for g in ps.stdout.splitlines() if g.strip() and "NVIDIA" not in g]
        except (OSError, subprocess.SubprocessError):   # a cold laptop can take longer than 20 s to answer WMI
            specs["gpu_query_failed"] = True
    smi = find_nvidia_smi()
    if smi:
        specs["nvidia"] = parse_nvidia_smi(run_cmd(
            [smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"]))
        if not specs["nvidia"]:   # very old drivers do not know compute_cap
            specs["nvidia"] = parse_nvidia_smi(run_cmd(
                [smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]))
    specs.update(classify_accelerator(specs))
    return specs


def _read(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _windows_ram_gb() -> float | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        stat = MemoryStatus()
        stat.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return round(stat.ullTotalPhys / 1e9, 1)
    except Exception:  # noqa: BLE001
        return None


def classify_accelerator(specs: dict) -> dict:
    """Pure: which torch device we will use and how much memory it has. Testable with fake specs."""
    out = {"accelerator": "cpu", "device_mem_gb": specs.get("ram_gb"), "bf16": False, "torch_backend": "cpu",
           "accelerator_note": ""}
    nvidia = specs.get("nvidia") or []
    if nvidia:
        gpu = nvidia[0]
        driver = parse_driver_version(gpu.get("driver", ""))
        if driver >= MIN_NVIDIA_DRIVER:
            out.update(accelerator="cuda", device_mem_gb=gpu.get("vram_gb"), torch_backend=TORCH_BACKEND_NVIDIA,
                       bf16=(gpu.get("compute_cap") or 0) >= BF16_MIN_COMPUTE_CAP)
        else:
            out["accelerator_note"] = "driver_too_old"
        return out
    if specs.get("system") == "Darwin":
        if specs.get("rosetta"):
            out["accelerator_note"] = "rosetta"
        elif specs.get("arch", "").startswith("arm") and tuple(specs.get("macos_version", (0, 0))) >= MIN_MACOS_FOR_MPS:
            out.update(accelerator="mps", device_mem_gb=specs.get("ram_gb"), torch_backend="default", bf16=True)
        elif specs.get("arch", "").startswith("arm"):
            out["accelerator_note"] = "macos_too_old"
        else:
            out["accelerator_note"] = "intel_mac"
    elif specs.get("other_gpus"):
        out["accelerator_note"] = "non_nvidia_gpu"
    return out


def hard_floors(specs: dict) -> list[dict]:
    """Pure. Only floors that make the whole run pointless. Each carries the arithmetic."""
    floors = []
    if specs.get("disk_free_gb") is not None and specs["disk_free_gb"] < MIN_FREE_DISK_GB:
        floors.append({"key": "disk", "have": specs["disk_free_gb"], "need": MIN_FREE_DISK_GB,
                       "zh": f"磁盘剩余 {specs['disk_free_gb']} GB < 需要 {MIN_FREE_DISK_GB} GB（环境约 7 GB + 权重约 13 GB + 数据与余量）。清出空间后重跑。",
                       "en": f"Free disk {specs['disk_free_gb']} GB < required {MIN_FREE_DISK_GB} GB (env ~7 GB + weights ~13 GB + data + headroom). Free space and rerun."})
    if specs.get("ram_gb") is not None and specs["ram_gb"] < MIN_RAM_GB:
        floors.append({"key": "ram", "have": specs["ram_gb"], "need": MIN_RAM_GB,
                       "zh": f"内存 {specs['ram_gb']} GB < 需要 {MIN_RAM_GB} GB。PyTorch 本身加任何一个模型都装不进去。",
                       "en": f"RAM {specs['ram_gb']} GB < required {MIN_RAM_GB} GB. PyTorch plus any model will not fit."})
    if specs.get("system") == "Windows":
        rel = str(specs.get("windows_release", ""))
        if rel and rel.split(".")[0].isdigit() and int(rel.split(".")[0]) < 10:
            floors.append({"key": "windows", "have": rel, "need": "10",
                           "zh": f"Windows {rel} 低于 PyTorch 支持的 Windows 10。",
                           "en": f"Windows {rel} is below Windows 10, the PyTorch wheel floor."})
    return floors


def estimate_table(specs: dict) -> dict:
    """Pure. Pre-install guess per level from the HF hardware guide. Labelled 'estimate' everywhere."""
    mem = specs.get("device_mem_gb") or 0
    acc = specs.get("accelerator", "cpu")
    out = {}
    for lv in LEVELS:
        low, high = ESTIMATE_VRAM_GB[lv.group]
        if acc == "cpu":
            train = "cloud"
        elif mem >= high:
            train = "local"
        elif mem >= low:
            train = "tight"
        else:
            train = "cloud"
        out[lv.id] = {"group": lv.group, "vram_bs8_gb": [low, high], "train_estimate": train}
    return out


def weight_floor(level: Level, params: int | None, dtype: str, device_mem_gb: float | None,
                 ram_gb: float | None = None) -> dict | None:
    """Pure. The only per-level floor: the weights alone, at the dtype lerobot loads them in, do not
    fit the device memory (or the RAM they pass through first)."""
    if not params:
        return None
    load_dtype = level.load_dtype or dtype
    need_gb = params * BYTES_PER_PARAM[load_dtype] / 1e9
    for what_zh, what_en, have in (("设备内存", "device memory", device_mem_gb), ("内存", "RAM", ram_gb)):
        if have and need_gb > have:
            return {"need_gb": round(need_gb, 1), "have_gb": have, "dtype": load_dtype,
                    "zh": f"权重 {need_gb:.1f} GB（{params/1e9:.2f}B 参数 × {load_dtype}，lerobot 加载时的精度）> {what_zh} {have} GB",
                    "en": f"weights {need_gb:.1f} GB ({params/1e9:.2f}B params x {load_dtype}, the dtype lerobot loads them in) > {what_en} {have} GB"}
    return None


# ----------------------------------------------------------------------------------------------
# Hugging Face helpers (stdlib HTTP; work before the venv exists).
# ----------------------------------------------------------------------------------------------

def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def hf_get_json(path: str, timeout=HF_HTTP_TIMEOUT_S) -> dict | list | None:
    req = urllib.request.Request(f"{HF_ENDPOINT}{path}", headers={"User-Agent": f"lerobot-doctor/{TOOL_VERSION}"})
    tok = hf_token() if HF_ENDPOINT == HF_OFFICIAL_ENDPOINT else None   # a mirror never sees the token; these endpoints are public anyway
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    for _ in range(HF_HTTP_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            continue
    return None


def hf_model_summary(repo: str) -> dict | None:
    """Parameter count (by stored dtype) and total file bytes, from the model API."""
    info = hf_get_json(f"/api/models/{repo}?blobs=true")
    if not isinstance(info, dict):
        return None
    st = info.get("safetensors") or {}
    params = st.get("total") or sum((st.get("parameters") or {}).values()) or None
    size = sum(s.get("size") or 0 for s in info.get("siblings") or [])
    return {"repo": repo, "params": params, "bytes": size, "gated": info.get("gated", False)}


# ----------------------------------------------------------------------------------------------
# Stage A printing.
# ----------------------------------------------------------------------------------------------

# Why the run is on the CPU although the machine has some GPU. Printed in stage 1 and again in the
# final report, so the box a user shares says the same thing.
ACCELERATOR_NOTES = {
    "driver_too_old": ("NVIDIA 驱动低于 570.86：这次按 CPU 测，GPU 一列全部记「未测」。升级驱动后重跑。",
                       "NVIDIA driver below 570.86: testing on CPU; the GPU column is 'not tested'. Update the driver and rerun."),
    "rosetta": ("Python 跑在 Rosetta 下（x86 版），MPS 不可用。装 arm64 版 Python 后重跑。",
                "Python runs under Rosetta (x86 build); MPS unavailable. Install arm64 Python and rerun."),
    "macos_too_old": ("macOS 低于 12.3，MPS 不可用，按 CPU 测。", "macOS below 12.3: no MPS, testing on CPU."),
    "intel_mac": ("Intel Mac：无加速器，按 CPU 测；torchcodec 无轮子，LeRobot 会自动改用 pyav。",
                  "Intel Mac: no accelerator, testing on CPU; no torchcodec wheel, LeRobot falls back to pyav."),
    "non_nvidia_gpu": ("检测到非 NVIDIA 显卡（含核显）：PyTorch 在这台机器上不用它，所有模型按 CPU 测。",
                       "A non-NVIDIA GPU (integrated ones included) was found: PyTorch does not use it here; every model is tested on the CPU."),
}


def print_specs(con: Console, specs: dict):
    con.line(bi("系统体检 · 第一段（只读，不安装任何东西）", "System check, stage 1 (read-only, installs nothing)"))
    rows = [
        ("系统 OS", specs["os"] + (" · WSL" if specs.get("wsl") else "")),
        ("架构 Arch", specs["arch"]),
        ("处理器 CPU", f"{specs['cpu']} · {specs['cpu_count']} threads"),
        ("内存 RAM", f"{specs['ram_gb']} GB" if specs["ram_gb"] else "?"),
        ("磁盘剩余 Free disk", f"{specs['disk_free_gb']} GB"),
        ("Python", specs["python"]),
        ("ffmpeg", "yes" if specs["ffmpeg"] else "no"),   # the CLI binary; which video decoder torch ends up with is measured later
    ]
    for g in specs["nvidia"]:
        rows.append(("显卡 GPU (NVIDIA)", f"{g['name']} · {g['vram_gb']} GB · driver {g['driver']}"
                                          + (f" · compute {g['compute_cap']}" if g['compute_cap'] else "")))
    for g in specs["other_gpus"]:
        rows.append(("显卡 GPU (other)", g))
    if specs.get("gpu_query_failed"):
        rows.append(("显卡 GPU (other)", "未能查询 / could not query"))
    acc = specs["accelerator"]
    acc_text = {"cuda": f"CUDA ({specs['device_mem_gb']} GB VRAM, bf16 {'yes' if specs['bf16'] else 'no'})",
                "mps": f"Apple MPS ({specs['device_mem_gb']} GB unified memory)",
                "cpu": "CPU only"}[acc]
    rows.append(("测试用设备 Device for tests", acc_text))
    width = max(dwidth(r[0]) for r in rows)
    for k, v in rows:
        con.line(f"  {pad(k, width)}  {v}")
    if specs.get("accelerator_note") in ACCELERATOR_NOTES:
        con.item("warn", *ACCELERATOR_NOTES[specs["accelerator_note"]])
    if specs.get("wsl"):
        con.item("info", "WSL：GPU 一般可用；USB 串口要用 usbipd 转发进来。", "WSL: GPU usually works; USB serial needs usbipd forwarding.")


def print_estimate(con: Console, est: dict):
    con.line("")
    con.line(bi("预估（来自官方硬件指南的静态表，不是结论；下面实测会覆盖它）",
                "Estimate (static table from the HF hardware guide, not a verdict; measurements below override it)"))
    words = {"local": ("本地可训", "train locally"), "tight": ("勉强，可能要减 batch", "tight, may need a smaller batch"),
             "cloud": ("训练需上云", "training needs cloud")}
    for lv in LEVELS:
        e = est[lv.id]
        zh, en = words[e["train_estimate"]]
        con.line(f"  {lv.id} {pad(lv.label, 9)} {pad(lv.group, 9)} BS8 峰值显存 {e['vram_bs8_gb'][0]}–{e['vram_bs8_gb'][1]} GB  →  {bi(zh, en)}")


# ----------------------------------------------------------------------------------------------
# Stage B part 1: build the environment with uv, then re-exec inside it.
# ----------------------------------------------------------------------------------------------

def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    home = Path.home()
    dirs = [home / ".local" / "bin", home / ".cargo" / "bin"]
    if platform.system() == "Windows":
        dirs.append(Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local"))) / "Programs" / "uv")
    for d in dirs:
        for name in ("uv", "uv.exe"):
            if (d / name).exists():
                return str(d / name)
    return None


def stream_process(cmd, con: Console, log_path: Path, env=None, cwd=None, timeout=None, on_line=None) -> int:
    """Run a command, mirror its output to the console (and a log), keep the heartbeat alive.

    The deadline is checked every CHILD_POLL_S whether or not the child prints: a child that hangs in
    silence (a stalled download, a wedged CUDA init) is killed like a chatty one. Returns the exit code,
    or -999 on timeout; whatever the child had written by then, terminated or not, still reaches on_line."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(map(str, cmd))}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                bufsize=0, start_new_session=platform.system() != "Windows")
        q: queue.Queue = queue.Queue()

        def pump():   # the only thread that touches the pipe; a raw pipe read returns as soon as any bytes exist
            try:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    q.put(chunk)
            except (OSError, ValueError):   # the pipe was closed under us after a kill
                pass
            q.put(None)

        threading.Thread(target=pump, name="child-stdout", daemon=True).start()

        def deliver(seg: bytes, is_cr: bool):
            text = seg.decode("utf-8", errors="replace")
            log.write(text + "\n")
            if on_line and on_line(text, is_cr):
                return
            if is_cr:
                con.transient(text[-200:])
            elif text.strip():
                con.raw(text + "\n")

        def feed(buf: bytes) -> bytes:
            while True:
                m = re.search(rb"[\r\n]", buf)
                if not m:
                    return buf
                deliver(buf[:m.start()], buf[m.start():m.end()] == b"\r")
                buf = buf[m.end():]

        deadline = time.monotonic() + timeout if timeout else None
        buf, timed_out = b"", False
        try:
            while True:
                try:
                    chunk = q.get(timeout=CHILD_POLL_S)
                except queue.Empty:
                    chunk = b""
                if chunk is None:
                    break
                if chunk:
                    buf = feed(buf + chunk)
                if deadline is not None and time.monotonic() > deadline:
                    timed_out = True
                    break
            if timed_out:
                kill_tree(proc)
                while True:   # what the reader had already queued before the kill
                    try:
                        chunk = q.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:
                        break
                    buf = feed(buf + chunk)
        except BaseException:   # Ctrl-C: the child sits in its own session and would not hear the terminal's
            kill_tree(proc)
            _reap(proc)
            raise
        if buf.strip():
            deliver(buf, False)   # an unterminated last line (a @@progress cut by the kill) still counts
        if timed_out:
            log.write("\n[lerobot-doctor] TIMEOUT\n")
            _reap(proc)
            return -999
        rc = proc.wait()
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass
        return rc


def _reap(proc: subprocess.Popen):
    """After a kill: close our end of the pipe and wait, so no zombie and no ResourceWarning mid-report."""
    try:
        proc.stdout.close()
    except (OSError, ValueError):
        pass
    try:
        proc.wait(timeout=CHILD_WAIT_AFTER_KILL_S)
    except (OSError, subprocess.TimeoutExpired):
        pass


def kill_tree(proc: subprocess.Popen):
    """The child and everything it spawned. POSIX: the child was started in its own session, so its
    process group id is its pid. Windows: taskkill /T walks the tree."""
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def log_since(log_path: Path, offset: int) -> str:
    """The bytes a log gained after `offset`: a retry must be classified by its own output, not the
    previous attempt's (the file is appended across attempts and across runs)."""
    try:
        with open(log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def import_probe(py: Path) -> tuple[int, dict | None, str]:
    """Run IMPORT_PROBE with `py`: (returncode, the JSON it printed or None, stderr tail). The JSON is
    the last line that parses, because torch or a library may print after it."""
    try:
        probe = subprocess.run([str(py), "-c", IMPORT_PROBE], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return -1, None, f"import probe did not finish in {PROBE_TIMEOUT_S} s"
    except OSError as e:
        return -1, None, str(e)
    info = None
    for line in reversed((probe.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                info = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return probe.returncode, info, (probe.stderr or "")[-2000:]


def venv_python_version(py: Path) -> str | None:
    """'3.12' for the interpreter at `py`, None when it cannot even start."""
    try:
        out = subprocess.run([str(py), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_env(con: Console, specs: dict, report: dict, args) -> Path | None:
    """Create ~/lerobot-doctor/.venv and install lerobot 0.6.1. Returns the venv python or None."""
    venv = WORK_DIR / ".venv"
    py = venv_python(venv)
    logs = WORK_DIR / "logs"
    install = report.setdefault("install", {"status": "NOT_RUN"})
    uv = find_uv()
    if not uv:
        install.update(status="FAIL_CRASH", reason="uv not found")
        con.item("bad", "找不到 uv。请用 doctor.sh / doctor.ps1 启动，它会先装 uv。",
                 "uv not found. Start with doctor.sh / doctor.ps1, which installs uv first.")
        return None
    backend = specs["torch_backend"]
    log_path = logs / "install.log"
    if py.exists():
        have = venv_python_version(py)
        if have != PYTHON_VERSION:   # a venv left by an older tool version, or one whose Python is gone
            con.item("warn", f"已有环境的 Python 是 {have or '坏的'}，不是 {PYTHON_VERSION}：重建",
                     f"the existing env has Python {have or 'broken'}, not {PYTHON_VERSION}: rebuilding it")
            shutil.rmtree(venv, ignore_errors=True)
    if not py.exists():
        con.activity(bi("创建虚拟环境", "creating virtualenv"))
        rc = stream_process([uv, "venv", str(venv), "--python", PYTHON_VERSION, "--seed"], con, log_path,
                            timeout=UV_VENV_TIMEOUT_S)
        if rc != 0:
            # uv fetches Python from GitHub when the machine has no 3.12; that is the usual failure here,
            # and it is uv's, not LeRobot's and not the machine's.
            install.update(status="FAIL_CRASH", reason=f"uv venv exited {rc}", log=str(log_path),
                           hint="UV_PYTHON_INSTALL_MIRROR")
            return None
    spec = f"lerobot[{LEROBOT_EXTRAS}]=={LEROBOT_VERSION}"
    cmd = [uv, "pip", "install", "--python", str(py), spec, VISER_SPEC]
    if backend != "default":
        cmd += ["--torch-backend", backend]
    con.activity(bi("安装 LeRobot", "installing LeRobot"))
    t0 = time.monotonic()
    env = dict(os.environ, UV_HTTP_TIMEOUT=str(UV_HTTP_TIMEOUT_S))
    for attempt in range(1, INSTALL_ATTEMPTS + 1):
        offset = log_path.stat().st_size if log_path.exists() else 0
        rc = stream_process(cmd, con, log_path, env=env, timeout=UV_INSTALL_TIMEOUT_S)
        if rc == 0:
            break
        tail = log_since(log_path, offset)[-4000:].lower()   # this attempt's output only
        if attempt < INSTALL_ATTEMPTS and (rc == -999 or "timeout" in tail or "timed out" in tail or "connection" in tail):
            con.item("warn", f"下载超时，重试 {attempt}/{INSTALL_ATTEMPTS - 1}（已下好的包不重下）",
                     f"download timed out, retry {attempt}/{INSTALL_ATTEMPTS - 1} (finished packages are cached)")
            continue
        break
    install["seconds"] = round(time.monotonic() - t0)
    install["torch_backend"] = backend
    install["log"] = str(log_path)
    if rc != 0:
        install.update(status="FAIL", reason=f"uv pip install exited {rc}" if rc != -999 else f"uv pip install did not finish in {UV_INSTALL_TIMEOUT_S // 60} min")
        return None
    rc, info, err = import_probe(py)
    if rc != 0 or info is None:
        install.update(status="FAIL", reason="import failed" if rc != 0 else "import probe printed no result", stderr=err)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(err)
        return None
    install.update(status="PASS", **info)
    return py


IMPORT_PROBE = r"""
import json, importlib.metadata as md
import lerobot, torch, viser
out = {"lerobot": md.version("lerobot"), "torch": torch.__version__, "viser": md.version("viser"),
       "cuda": torch.cuda.is_available(), "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
       "cuda_version": torch.version.cuda, "torchcodec": None}
try:
    import torchcodec; out["torchcodec"] = md.version("torchcodec")
except Exception as e:
    out["torchcodec"] = None; out["torchcodec_error"] = str(e)[:200]
print(json.dumps(out))
"""


# ----------------------------------------------------------------------------------------------
# Ladder state machine (pure functions; unit-tested).
# ----------------------------------------------------------------------------------------------

def infer_precheck(level: Level, results: dict, specs: dict, weights: dict, dtype: str) -> dict | None:
    """Decide whether to *skip* the inference probe. Returns a result dict or None (= run it)."""
    # floor 1: weights alone do not fit. On a CPU "device memory" is the RAM: one check, one name.
    w = weights.get(level.id) or {}
    dev_mem = specs.get("device_mem_gb") if specs.get("accelerator") != "cpu" else None
    fl = weight_floor(level, w.get("params"), dtype, dev_mem, specs.get("ram_gb"))
    if fl:
        return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": "weights_exceed_memory", **fl}
    # floor 2: memory monotonic - a smaller level already failed on memory (same device, same dtype)
    my_params = w.get("params") or 0
    for other in LEVELS:
        if other.id == level.id:
            continue
        r = results.get(other.id, {}).get("infer") or {}
        if r.get("status") in ("FAIL_OOM", "FAIL_RAM") and ((weights.get(other.id) or {}).get("params") or 0) <= my_params \
                and r.get("dtype", dtype) == dtype and my_params:
            return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": f"smaller_level_oom:{other.id}",
                    "zh": f"{other.id} {other.label} 已装不下（{r['status']}），本级权重更大",
                    "en": f"{other.id} {other.label} already did not fit ({r['status']}); this level's weights are larger"}
    return None


def train_precheck(level: Level, results: dict) -> dict | None:
    """Decide whether to skip the training probe based on this level's inference result."""
    inf = results.get(level.id, {}).get("infer") or {"status": "NOT_RUN"}
    st = inf.get("status")
    if st in MEMORY_FAILS:
        return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": f"infer_{st}",
                "zh": "前向都装不下，反向必然装不下", "en": "the forward pass did not fit; the backward pass cannot"}
    if st in STATUS_NOT_RUN:
        return {"status": "NOT_RUN", "evidence": "not_run", "reason": f"infer_{st}",
                "zh": f"推理阶段未测（{inf.get('zh', st)}），同一前提", "en": f"inference was not run ({inf.get('en', st)}); same precondition"}
    smaller = [other for other in LEVELS if LEVELS.index(other) < LEVELS.index(level)]
    for other in smaller:   # memory monotonic for training too
        r = results.get(other.id, {}).get("train") or {}
        if r.get("status") in ("FAIL_OOM", "FAIL_RAM") and (r.get("params") or 0) <= (inf.get("params") or 0):
            return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": f"smaller_level_oom:{other.id}",
                    "zh": f"{other.id} {other.label} batch 1 都训不了，本级更大", "en": f"{other.id} {other.label} failed even at batch 1; this level is larger"}
    for other in smaller:   # speed monotonic on a CPU/MPS: a smaller level that already needs the cloud settles the larger ones
        r = results.get(other.id, {}).get("train") or {}
        if r.get("device") in ("cpu", "mps") and r.get("status") in ("PASS", "TIMEOUT") and r.get("hours") is not None \
                and (r["hours"] > OVERNIGHT_H or r.get("batch") == 1):
            return {"status": "SKIPPED_SLOWER", "evidence": "floor", "reason": f"slower_level_cloud:{other.id}",
                    "zh": f"{other.id} {other.label} 在这颗 {r['device'].upper()} 上已经要上云（约 {r['hours']:.0f} 小时），本级更大只会更慢；推理照测，训练不测",
                    "en": f"{other.id} {other.label} already needs the cloud on this {r['device'].upper()} (~{r['hours']:.0f} h); this level is larger and only slower. Inference is still measured, training is not"}
    return None


def classify_exit(returncode: int, log_text: str) -> str:
    """Map a dead child to a status from its exit code and log."""
    low = log_text.lower()
    if "out of memory" in low or "outofmemoryerror" in low or "mps backend out of memory" in low:
        return "FAIL_OOM"
    if returncode in (-9, 137):   # SIGKILL: the OOM killer, on Linux, is the usual sender
        return "FAIL_RAM"
    if returncode in (3221225477, -1073741819):   # 0xC0000005 access violation: a DLL/driver/CPU-flag problem of the stack, not memory
        return "FAIL_CRASH"
    if "is required but not installed" in low or "no module named" in low:
        return "FAIL_DEP"
    if "gated" in low or "401 client error" in low or "403 client error" in low:
        return "BLOCKED_GATED"
    if any(k in low for k in ("connectionerror", "max retries", "name resolution", "timed out", "httperror", "couldn't connect",
                              "cas client error", "reconstruction error", "decoding response body", "incompleteread",
                              "readtimeout", "chunkedencodingerror", "remote end closed")):
        return "FAIL_DOWNLOAD"
    return "FAIL_CRASH"


def realtime_status(latency_s: float, n_action_steps: int) -> tuple[str, float]:
    budget = n_action_steps / CONTROL_FPS
    if latency_s <= REALTIME_RATIO * budget:
        return "PASS", budget
    if latency_s <= MARGINAL_RATIO * budget:
        return "MARGINAL", budget
    return "TOO_SLOW", budget


def projected_hours(update_s: float, batch: int) -> float:
    steps = REF_EPOCHS * REF_FRAMES / batch
    return steps * update_s / 3600


def infer_summary(lat: list[float], n_action_steps: int) -> dict:
    """Pure. The band is decided on p95: the arm waits for the slowest chunk, not the typical one.
    The median stays in the JSON and the table."""
    med, p95 = statistics.median(lat), percentile(lat, 0.95)
    status, budget = realtime_status(p95, n_action_steps)
    return {"status": status, "latency_ms": round(med * 1000, 1), "p95_ms": round(p95 * 1000, 1),
            "budget_ms": round(budget * 1000), "timed_calls": len(lat)}


def demo_summary(done: int, n_avail: int, elapsed_s: float, refill_lat: list[float]) -> dict:
    """Pure. PASS = every step and at least DEMO_OK_RATIO of the target rate; SLOW = finished but
    below it (the arm stalls at every chunk refill); TOO_SLOW = cut off by the wall-clock cap."""
    hz = done / elapsed_s if elapsed_s > 0 else 0.0
    if done < n_avail:
        status = "TOO_SLOW"
    elif hz >= DEMO_OK_RATIO * CONTROL_FPS:
        status = "PASS"
    else:
        status = "SLOW"
    return {"status": status, "steps": done, "of_steps": n_avail, "seconds": round(elapsed_s, 1), "hz": round(hz, 1),
            "target_hz": CONTROL_FPS, "refill_ms": round(statistics.median(refill_lat) * 1000, 1) if refill_lat else None}


def demo_text(demo: dict | None) -> tuple[str, str]:
    """(zh, en) clause about the simulated task, from measured numbers only; ("", "") when it did not run."""
    if not demo or not demo.get("status"):
        return "", ""
    st, hz = demo["status"], demo.get("hz") or 0
    s, steps, tgt = demo.get("seconds") or 0, demo.get("steps") or 0, demo.get("target_hz") or CONTROL_FPS
    stall = demo.get("refill_ms")
    stall_zh = f"，每块等 {stall:.0f} ms" if stall else ""
    stall_en = f", {stall:.0f} ms wait per chunk" if stall else ""
    if st == "PASS":
        return (f"；模拟执行 {s:.1f} s 通过，控制 {hz:.0f} Hz", f"; simulated task {s:.1f} s OK at {hz:.0f} Hz")
    if st == "SLOW":
        return (f"；模拟执行跑完但只有 {hz:.0f} Hz（目标 {tgt}），用了 {s:.1f} s{stall_zh}",
                f"; simulated task finished but only {hz:.0f} Hz (target {tgt}), took {s:.1f} s{stall_en}")
    if st == "SKIPPED":
        return "", ""
    return (f"；模拟执行 {s:.0f} s 内只走了 {steps} 步（{hz:.0f} Hz）{stall_zh}",
            f"; simulated task: only {steps} steps in {s:.0f} s ({hz:.0f} Hz){stall_en}")


def partial_train_numbers(r: dict) -> dict | None:
    """Pure. A training probe that hit its budget still reported every timed step through @@progress:
    {update_s, batch, hours, n} from those, or None when no step beyond the warm-up finished."""
    p = r.get("partial") or {}
    times, b = p.get("step_times") or [], p.get("batch")
    if not times or not b:
        return None
    s = statistics.median(times)
    return {"update_s": round(s, 3), "batch": b, "hours": round(projected_hours(s, b), 1), "n": len(times)}


# ----------------------------------------------------------------------------------------------
# Verdicts (pure; unit-tested).
# ----------------------------------------------------------------------------------------------

def _small_sample(r: dict) -> tuple[str, str]:
    n = r.get("timed_calls")
    return (f"（n={n}）", f" (n={n})") if n and n < TIMED_CPU else ("", "")


def _budget_text(seconds) -> tuple[str, str]:
    """'15 分钟' / '15 min' for a probe budget; seconds below a minute stay seconds."""
    s = seconds or 0
    if s >= 60:
        return f"{s / 60:.0f} 分钟", f"{s / 60:.0f} min"
    return f"{s:.0f} 秒", f"{s:.0f} s"


def infer_verdict(level: Level, r: dict) -> dict:
    st = r.get("status", "NOT_RUN")
    ms = r.get("latency_ms") or 0
    bud = r.get("budget_ms") or 0
    p95 = r.get("p95_ms")
    lat_zh = f"每块 {ms:.0f} ms" + (f"，p95 {p95:.0f}" if p95 and round(p95) != round(ms) else "") + f"，预算 {bud:.0f} ms"
    lat_en = f"{ms:.0f} ms per chunk" + (f", p95 {p95:.0f}" if p95 and round(p95) != round(ms) else "") + f", budget {bud:.0f} ms"
    n_zh, n_en = _small_sample(r)
    d_zh, d_en = demo_text(r.get("demo"))
    if st == "PASS":
        return {"mark": "ok", "evidence": "measured",
                "zh": f"本地实时推理（{lat_zh}）{n_zh}{d_zh}", "en": f"real-time local inference ({lat_en}){n_en}{d_en}"}
    if st == "MARGINAL":
        return {"mark": "warn", "evidence": "measured",
                "zh": f"本地可推理但接近上限（{lat_zh}）{n_zh}；建议相机降到 320×240 或加长 chunk{d_zh}",
                "en": f"local inference works but near the limit ({lat_en}){n_en}; lower cameras to 320x240 or lengthen the chunk{d_en}"}
    if st == "TOO_SLOW":
        return {"mark": "bad", "evidence": "measured",
                "zh": f"本地推理达不到实时（{lat_zh}）{n_zh}", "en": f"local inference is not real-time ({lat_en}){n_en}"}
    if st in ("FAIL_OOM", "FAIL_RAM"):
        what = "显存" if r.get("device") == "cuda" else "内存"
        return {"mark": "bad", "evidence": "measured",
                "zh": f"本地装不下（加载时{what}耗尽）", "en": f"does not fit locally (ran out of {'VRAM' if r.get('device') == 'cuda' else 'memory'} while loading)"}
    if st == "SKIPPED_FLOOR":
        return {"mark": "bad", "evidence": "floor", "zh": f"本地装不下：{r.get('zh', '')}", "en": f"does not fit locally: {r.get('en', '')}"}
    # "see log" is useless without the log's path and the line that failed; run_worker recorded both.
    err, log = (r.get("error") or "")[:200], r.get("log") or ""
    seen_zh = (f"（{err}）" if err else "") + (f"，日志 {log}" if log else "")
    seen_en = (f" ({err})" if err else "") + (f", log {log}" if log else "")
    phase_zh, phase_en = split_bi((r.get("partial") or {}).get("phase") or "")
    b_zh, b_en = _budget_text(r.get("seconds"))
    within_zh = f"{b_zh}内" if r.get("seconds") else "预算时间内"
    within_en = f"in {b_en}" if r.get("seconds") else "within its budget"
    reasons = {
        "BLOCKED_GATED": ("未测：这个模型的仓库要先在 Hugging Face 上同意许可并登录", "not tested: this model's repo needs a Hugging Face login and license acceptance"),
        "FAIL_DEP": (f"未测：依赖没装上{seen_zh}", f"not tested: a dependency is missing{seen_en}"),
        "FAIL_DOWNLOAD": (f"未测：下载失败（网络）{seen_zh}", f"not tested: download failed (network){seen_en}"),
        "TIMEOUT": (f"未测：{within_zh}没跑完；最后在做：{phase_zh or '—'}{seen_zh}", f"not tested: did not finish {within_en}; last seen: {phase_en or '-'}{seen_en}"),
        "FAIL_CRASH": (f"未测：程序异常{seen_zh}。这是工具或环境的问题，不是你电脑的结论", f"not tested: the probe crashed{seen_en}. That is a tool/environment problem, not a verdict about your machine"),
        "NOT_RUN": (r.get("zh", "未测"), r.get("en", "not tested")),
    }
    zh, en = reasons.get(st, reasons["NOT_RUN"])
    return {"mark": "skip", "evidence": "not_run", "zh": zh, "en": en}


def _train_bands(b: int, s: float, h: float) -> dict:
    """The local / overnight / cloud sentence for one measured (or projected) step time."""
    base_zh = f"batch {b}，每步 {s:.2f} s，参考任务约 {h:.1f} 小时"
    base_en = f"batch {b}, {s:.2f} s per step, ~{h:.1f} h for the reference task"
    if h <= LOCAL_OK_H:
        v = {"mark": "ok", "zh": f"本地训练（{base_zh}）", "en": f"train locally ({base_en})", "cloud": False}
    elif h <= OVERNIGHT_H:
        v = {"mark": "warn", "zh": f"本地可训练，过一夜（{base_zh}）", "en": f"trainable locally overnight ({base_en})", "cloud": False}
    else:
        v = {"mark": "warn", "zh": f"本地能训但太慢（{base_zh}）→ 建议上云", "en": f"trainable locally but too slow ({base_en}) -> cloud recommended", "cloud": True}
    if b == 1:
        v["zh"] += "；batch 只能 1，效果打折 → 建议上云"
        v["en"] += "; batch 1 only, quality suffers -> cloud recommended"
        v["cloud"] = True
        v["mark"] = "warn"
    return v


def train_verdict(level: Level, r: dict) -> dict:
    st = r.get("status", "NOT_RUN")
    if st == "PASS":
        if any(r.get(k) is None for k in ("batch", "update_s", "hours")):   # a PASS with no numbers is a broken record, not a verdict
            return {"mark": "skip", "evidence": "not_run", "cloud": None,
                    "zh": "未测：训练结果缺少实测数字（记录不完整）", "en": "not tested: the training record has no measured numbers (incomplete record)"}
        v = _train_bands(r["batch"], r["update_s"], r["hours"])
        v["evidence"] = "measured"
        return v
    if st == "TIMEOUT":
        b_zh, b_en = _budget_text(r.get("seconds"))
        pn = partial_train_numbers(r)
        if pn:
            v = _train_bands(pn["batch"], pn["update_s"], pn["hours"])
            v["zh"] += f"（据 {pn['n']} 步推算，{b_zh}预算用完即停）"
            v["en"] += f" (projected from {pn['n']} steps; the run stopped at its {b_en} budget)"
            v["evidence"] = "partial"
            return v
        phase_zh, phase_en = split_bi((r.get("partial") or {}).get("phase") or "")
        n_steps = TRAIN_WARMUP + TRAIN_TIMED
        within_zh = f"{b_zh}内" if r.get("seconds") else "预算时间内"
        within_en = f"in {b_en}" if r.get("seconds") else "within its budget"
        return {"mark": "skip", "evidence": "not_run", "cloud": None,
                "zh": f"未测：{within_zh}没跑完 {n_steps} 步；最后在做：{phase_zh or '—'}",
                "en": f"not tested: did not finish {n_steps} steps {within_en}; last seen: {phase_en or '-'}"}
    if st in ("FAIL_OOM", "FAIL_RAM"):
        return {"mark": "bad", "evidence": "measured", "cloud": True,
                "zh": "本地训不了（batch 1 也装不下）→ 需要上云", "en": "cannot train locally (even batch 1 does not fit) -> needs cloud"}
    if st == "SKIPPED_FLOOR":
        return {"mark": "bad", "evidence": "floor", "cloud": True,
                "zh": f"本地训不了：{r.get('zh', '')}", "en": f"cannot train locally: {r.get('en', '')}"}
    if st == "SKIPPED_SLOWER":
        return {"mark": "warn", "evidence": "floor", "cloud": True,
                "zh": f"建议上云：{r.get('zh', '')}", "en": f"cloud recommended: {r.get('en', '')}"}
    v = infer_verdict(level, r)
    v["cloud"] = None
    return v


def route_verdict(levels_v: dict) -> dict:
    """R1..R5 from the plan. levels_v: {level_id: {"infer": verdict, "train": verdict}}."""
    def infer_ok(v):
        return v["infer"]["mark"] in ("ok", "warn") and v["infer"]["evidence"] == "measured"

    def train_local(v):
        return v["train"]["mark"] in ("ok", "warn") and v["train"].get("cloud") is False

    def train_cloud(v):
        return v["train"].get("cloud") is True

    ordered = [lv for lv in LEVELS if lv.id in levels_v]
    r1 = [lv for lv in ordered if infer_ok(levels_v[lv.id]) and train_local(levels_v[lv.id])]
    if r1:
        k = r1[-1]
        return {"rule": "R1", "level": k.id,
                "zh": f"全流程本地：录数据 → 本地训练 {k.label} → 本地运行。",
                "en": f"Everything local: record data -> train {k.label} locally -> run locally."}
    any_infer_ok = [lv for lv in ordered if infer_ok(levels_v[lv.id])]
    r2 = [lv for lv in ordered if infer_ok(levels_v[lv.id]) and train_cloud(levels_v[lv.id])]
    if r2:
        k = r2[-1]
        top = any_infer_ok[-1]   # a higher level may run locally too, with its training verdict still open
        more_zh = f" 推理最高可到 {top.label}（它的训练结论未定）。" if top is not k else ""
        more_en = f" Inference runs locally up to {top.label} (its training verdict is open)." if top is not k else ""
        return {"rule": "R2", "level": k.id,
                "zh": f"录数据在本地 → 上云训练 {k.label} → 权重拿回本地推理。{more_zh}",
                "en": f"Record locally -> train {k.label} in the cloud -> bring the weights back and run locally.{more_en}"}
    if any_infer_ok:
        k = any_infer_ok[-1]
        return {"rule": "R3", "level": k.id,
                "zh": f"本地能跑 {k.label} 推理；训练结果未定（见各级）。先按「上云训练 + 本地推理」规划。",
                "en": f"{k.label} inference runs locally; training verdict pending (see levels). Plan for cloud training + local inference."}
    all_bad = all(levels_v[lv.id]["infer"]["mark"] == "bad" for lv in ordered) if ordered else False
    if all_bad:
        return {"rule": "R4", "level": None,
                "zh": "这台机器能做组装 / 标定 / 遥操作 / 录数据；训练与推理都要另找机器或上云。",
                "en": "This machine can assemble / calibrate / teleoperate / record; training and inference need another machine or the cloud."}
    first = next((levels_v[lv.id]["infer"] for lv in ordered if levels_v[lv.id]["infer"]["mark"] == "skip"), None)
    return {"rule": "R5", "level": None,
            "zh": "硬件结论未得出：" + (first["zh"] if first else "没有任何级别完成测试") + "。处理后重跑。",
            "en": "No hardware verdict: " + (first["en"] if first else "no level completed") + ". Fix and rerun."}


def basics_notes(specs: dict, install: dict) -> list[tuple[str, str]]:
    notes = []
    if specs.get("accelerator_note") in ACCELERATOR_NOTES:   # the box a user shares must say why the CPU did the work
        notes.append(ACCELERATOR_NOTES[specs["accelerator_note"]])
    if specs["system"] == "Darwin":
        notes.append(("键盘遥操作要给终端「辅助功能」权限；课程的相机脚本是 Linux 专用，mac 上用 lerobot-find-cameras。",
                      "Keyboard teleop needs Accessibility permission for the terminal; the course camera script is Linux-only, use lerobot-find-cameras on mac."))
    if specs["system"] == "Windows":
        notes.append(("串口叫 COMx，不是 /dev/ttyACM0；课程命令里替换即可。", "Serial ports are COMx, not /dev/ttyACM0; substitute in the course commands."))
    if specs.get("wsl"):
        notes.append(("USB 串口要用 usbipd 转发进 WSL。", "USB serial must be forwarded into WSL with usbipd."))
    if install.get("torchcodec") is None and install.get("status") == "PASS":
        notes.append(("torchcodec 不可用，视频解码走 pyav：不影响任何结论，只让下载后的准备阶段慢一些。",
                      "torchcodec unavailable, video decoding uses pyav: no verdict depends on it; only the preparation after downloads is slower."))
    if install.get("version_mismatch"):
        notes.append((f"注意：这次跑的是已有环境里的 lerobot {install['version_mismatch']}，不是 {LEROBOT_VERSION}（--skip-install）。",
                      f"Note: this run used lerobot {install['version_mismatch']} from the existing env, not {LEROBOT_VERSION} (--skip-install)."))
    return notes


def install_failure_text(install: dict) -> tuple[str, str]:
    """What to call a failed stage 2. uv failing to build the environment is not 'LeRobot did not install'."""
    reason, log = install.get("reason", ""), install.get("log", "")
    if install.get("status") == "FAIL_CRASH":
        hint_zh = "；防火墙后先设 UV_PYTHON_INSTALL_MIRROR 再重跑" if install.get("hint") == "UV_PYTHON_INSTALL_MIRROR" else ""
        hint_en = "; behind a firewall set UV_PYTHON_INSTALL_MIRROR and rerun" if install.get("hint") == "UV_PYTHON_INSTALL_MIRROR" else ""
        return (f"uv 没能建起环境（{reason}）：这是工具或网络的问题，不是 LeRobot 装不上，也不是你电脑的结论{hint_zh}。日志 {log}",
                f"uv could not build the environment ({reason}): a tool/network problem, not LeRobot failing and not a verdict about your machine{hint_en}. Log {log}")
    return (f"LeRobot {LEROBOT_VERSION} 装不上：{reason}。日志 {log}", f"LeRobot {LEROBOT_VERSION} did not install: {reason}. Log {log}")


def evaluate(report: dict) -> dict:
    """Turn raw results into verdicts. Pure."""
    install = report.get("install", {})
    out = {"install": install.get("status"), "levels": {}, "basics": None, "route": None, "notes": []}
    if install.get("status") != "PASS":
        zh, en = install_failure_text(install)
        out["route"] = {"rule": "R0", "zh": zh, "en": en}
        return out
    ds = report.get("dataset", {})
    if ds.get("status") != "PASS":
        # worker_dataset downloads AND decodes frame 0: a network failure, a broken video path, a full
        # disk and a timeout all land here, and only the status tells them apart.
        err = ds.get("error") or ""
        st = ds.get("status", "NOT_RUN")
        if st == "FAIL_DOWNLOAD":
            zh, en = "样例数据下载失败（网络）", "Sample dataset download failed (network)"
        elif st == "TIMEOUT":
            zh, en = "样例数据在预算时间内没准备好（网络或磁盘很慢）", "Sample dataset was not ready within its budget (slow network or disk)"
        else:
            zh, en = f"样例数据准备失败（{st}{'：' + err if err else ''}）", f"Sample dataset preparation failed ({st}{': ' + err if err else ''})"
        out["route"] = {"rule": "R0", "zh": f"{zh}，硬件没有得出任何结论。", "en": f"{en}; no hardware verdict."}
        out["basics"] = {"mark": "skip"}
        return out
    out["basics"] = {"mark": "skip"}   # assemble / calibrate / teleoperate / record: not measured by this tool
    out["notes"] = basics_notes(report["specs"], install)
    for lv in LEVELS:
        res = report.get("levels", {}).get(lv.id, {})
        out["levels"][lv.id] = {"infer": infer_verdict(lv, res.get("infer") or {"status": "NOT_RUN"}),
                                "train": train_verdict(lv, res.get("train") or {"status": "NOT_RUN"})}
    out["route"] = route_verdict(out["levels"])
    return out


# ----------------------------------------------------------------------------------------------
# The boxed terminal report.
# ----------------------------------------------------------------------------------------------

REPORT_MAX_WIDTH = 100
REPORT_MIN_WIDTH = 60
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

BOX = {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║", "ml": "╠", "mr": "╣",
       "rule": "─", "col": "│", "tee_l": "├", "tee_r": "┤", "cross": "┼", "tee_t": "┬", "tee_b": "┴",
       "ctl": "┌", "ctr": "┐", "cbl": "└", "cbr": "┘"}
BOX_ASCII = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "=", "v": "|", "ml": "+", "mr": "+",
             "rule": "-", "col": "|", "tee_l": "+", "tee_r": "+", "cross": "+", "tee_t": "+", "tee_b": "+",
             "ctl": "+", "ctr": "+", "cbl": "+", "cbr": "+"}
# Single-cell marks: emoji are two cells wide in some terminals and one in others, which breaks
# a box's right border. Colour carries the meaning on a terminal; the glyph alone does in a log.
MARKS = {"ok": "✓", "warn": "!", "bad": "✗", "skip": "–"}
MARKS_ASCII = {"ok": "+", "warn": "!", "bad": "x", "skip": "-"}
COLORS = {"ok": "\x1b[32m", "warn": "\x1b[33m", "bad": "\x1b[31m", "skip": "\x1b[2m",
          "bold": "\x1b[1m", "dim": "\x1b[2m", "reset": "\x1b[0m"}


def vis_width(text: str) -> int:
    return dwidth(_ANSI_RE.sub("", text))


def wrap_cells(text: str, width: int) -> list[str]:
    """Wrap by display width, preferring spaces; CJK text can break anywhere."""
    words, lines, cur = text.split(" "), [], ""
    for word in words:
        cand = word if not cur else cur + " " + word
        if vis_width(cand) <= width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        cur = ""
        while vis_width(word) > width:          # a single over-long token (CJK run): cut it
            piece, used = "", 0
            for ch in word:
                if used + dwidth(ch) > width:
                    break
                piece += ch
                used += dwidth(ch)
            lines.append(piece)
            word = word[len(piece):]
        cur = word
    if cur or not lines:
        lines.append(cur)
    return lines


class BoxWriter:
    """Draws one bordered report. Every line is padded to the same visible width."""

    def __init__(self, con: Console, width: int):
        self.con = con
        self.chars = BOX_ASCII if con.ascii else BOX
        self.marks = MARKS_ASCII if con.ascii else MARKS
        self.color = con.tty and not con.ascii
        self.width = width                     # total width including the two border cells
        self.inner = width - 2

    def paint(self, kind: str, text: str) -> str:
        return f"{COLORS[kind]}{text}{COLORS['reset']}" if self.color else text

    def mark(self, kind: str) -> str:
        return self.paint(kind, self.marks[kind])

    def top(self):
        self.con.line(self.chars["tl"] + self.chars["h"] * self.inner + self.chars["tr"])

    def bottom(self):
        self.con.line(self.chars["bl"] + self.chars["h"] * self.inner + self.chars["br"])

    def divider(self):
        self.con.line(self.chars["ml"] + self.chars["h"] * self.inner + self.chars["mr"])

    def row(self, text: str = "", indent: int = 2):
        for piece in wrap_cells(text, self.inner - indent - 1) if text else [""]:
            pad = self.inner - indent - vis_width(piece)
            self.con.line(f"{self.chars['v']}{' ' * indent}{piece}{' ' * max(pad, 0)}{self.chars['v']}")

    def heading(self, text: str):
        self.row(self.paint("bold", text), indent=2)

    def table(self, header: list[str], rows: list[list[str]], indent: int = 2):
        """A light table inside the box; column widths from content, last column absorbs the rest."""
        cols = len(header)
        widths = [max(vis_width(r[i]) for r in [header] + rows) for i in range(cols)]
        avail = self.inner - indent - 1 - (3 * (cols - 1)) - 4
        if sum(widths) > avail:                 # squeeze the two verdict columns evenly
            extra = sum(widths) - avail
            for i in (cols - 1, cols - 2):
                cut = min(extra, widths[i] - 12)
                widths[i] -= max(cut, 0)
                extra -= max(cut, 0)
        c = self.chars

        def fmt(cells):
            out = []
            for i, cell in enumerate(cells):
                lines = wrap_cells(cell, widths[i])
                out.append(lines)
            height = max(len(x) for x in out)
            for k in range(height):
                parts = []
                for i in range(cols):
                    piece = out[i][k] if k < len(out[i]) else ""
                    parts.append(piece + " " * (widths[i] - vis_width(piece)))
                self.row(f"{c['col']} " + f" {c['col']} ".join(parts) + f" {c['col']}", indent=indent)

        self.row(c["ctl"] + c["tee_t"].join(c["rule"] * (w + 2) for w in widths) + c["ctr"], indent=indent)
        fmt(header)
        self.row(c["tee_l"] + c["cross"].join(c["rule"] * (w + 2) for w in widths) + c["tee_r"], indent=indent)
        for r in rows:
            fmt(r)
        self.row(c["cbl"] + c["tee_b"].join(c["rule"] * (w + 2) for w in widths) + c["cbr"], indent=indent)


# One dictionary per language. The report is printed twice, a complete English box first, then a
# complete Chinese one: mixed-language cells made both halves hard to scan. Verdict sentences already
# arrive as {"zh": ..., "en": ...} pairs; this table covers the fixed words of the box itself.
REPORT_TEXT = {
    "en": {
        "title": "Report", "machine": "Machine", "not_used": "not used this run", "video": "video decode",
        "not_installed": "did not install", "basics": "Basics",
        "basics_line": "assemble / calibrate / teleoperate / record: not tested here (they need only USB and a serial port)",
        "gpu_unused": "not used by PyTorch",
        "levels": "Levels", "header": ["Lv", "Model", "Inference", "Training"],
        "legend": "{ok} ok   {warn} conditional   {bad} failed / floor   {skip} not tested",
        "details": "Details and evidence", "infer": "inference", "train": "training", "estimate": "estimate",
        "sep": ": ", "lp": " (", "rp": ")",
        "est_words": {"local": "train locally", "tight": "tight", "cloud": "cloud"},
        "est_note": "the estimate said '{w}'; the measurement wins", "route": "SO-101 route",
        "per_chunk": "ms/chunk", "budget": "budget", "too_slow": "too slow", "slow": "slow", "no_fit": "does not fit",
        "not_tested": "not tested", "to_cloud": " -> cloud", "overnight": " overnight",
        "no_fit_cloud": "does not fit -> cloud", "no_forward_cloud": "no forward pass -> cloud",
        "slower_cloud": "only slower -> cloud", "partial": "~",
        "after_training": "after {n} more training steps the simulated task's joint error went from {a}° to {b}°",
        "ladder": "batch ladder {seq}", "peak": "peak {gb} GB",
    },
    "zh": {
        "title": "体检报告", "machine": "电脑", "not_used": "本次未用", "video": "视频解码",
        "not_installed": "装不上", "basics": "基础",
        "basics_line": "组装 / 标定 / 遥操作 / 录数据：本工具未测（只需 USB 和串口）",
        "gpu_unused": "PyTorch 不用它",
        "levels": "各级实测", "header": ["级", "模型", "推理", "训练"],
        "legend": "{ok} 通过   {warn} 有条件   {bad} 失败 / 硬门槛   {skip} 未测",
        "details": "说明与依据", "infer": "推理", "train": "训练", "estimate": "预估",
        "sep": "：", "lp": "（", "rp": "）",
        "est_words": {"local": "本地可训", "tight": "勉强", "cloud": "需上云"},
        "est_note": "预估表曾说「{w}」，以实测为准", "route": "SO-101 路线",
        "per_chunk": "ms/块", "budget": "预算", "too_slow": "太慢", "slow": "偏慢", "no_fit": "装不下",
        "not_tested": "未测", "to_cloud": " → 上云", "overnight": " 过夜",
        "no_fit_cloud": "装不下 → 上云", "no_forward_cloud": "前向都装不下 → 上云",
        "slower_cloud": "只会更慢 → 上云", "partial": "约",
        "after_training": "再训 {n} 步后，模拟任务的关节误差从 {a}° 降到 {b}°",
        "ladder": "batch 阶梯 {seq}", "peak": "峰值 {gb} GB",
    },
}


def short_infer(v: dict, r: dict, t: dict) -> str:
    """One table cell for the inference verdict."""
    st = r.get("status", "NOT_RUN")
    ms = r.get("latency_ms") or 0
    demo = r.get("demo") or {}
    n = r.get("timed_calls")
    small = f" n={n}" if n and n < TIMED_CPU else ""
    hz = ""
    if demo.get("hz") is not None and demo.get("status") != "SKIPPED":
        hz = f" · {demo['hz']:.0f} Hz" + ("" if demo.get("status") == "PASS" else f" {t['slow']}")
    if st == "PASS":
        return f"{ms:.0f} {t['per_chunk']}{small}{hz}"
    if st == "MARGINAL":
        return f"{ms:.0f} {t['per_chunk']}{t['lp']}{t['budget']} {r.get('budget_ms') or 0:.0f}{t['rp']}{small}{hz}"
    if st == "TOO_SLOW":
        return f"{ms:.0f} {t['per_chunk']}{t['lp']}{t['budget']} {r.get('budget_ms') or 0:.0f}{t['rp']} {t['too_slow']}{small}"
    if st in ("FAIL_OOM", "FAIL_RAM", "SKIPPED_FLOOR"):
        return t["no_fit"]
    return t["not_tested"]


def short_train(v: dict, r: dict, t: dict) -> str:
    st = r.get("status", "NOT_RUN")
    if st in ("PASS", "TIMEOUT") and r.get("hours") is not None and r.get("batch") is not None:
        base = f"{t['partial'] if st == 'TIMEOUT' else ''}{r['hours']:.1f} h · batch {r['batch']}"
        if v.get("cloud"):
            return base + t["to_cloud"]
        if v.get("mark") == "warn":
            return base + t["overnight"]
        return base
    if st in ("FAIL_OOM", "FAIL_RAM"):
        return t["no_fit_cloud"]
    if st == "SKIPPED_FLOOR":
        return t["no_forward_cloud"]
    if st == "SKIPPED_SLOWER":
        return t["slower_cloud"]
    return t["not_tested"]


def ladder_text(r: dict, t: dict) -> str:
    """'batch ladder 8 OOM -> 4 OK · peak 9.8 GB' for a training result that stepped down, else ''."""
    attempts = r.get("attempts") or []
    if not any(a.get("status") == "FAIL_OOM" for a in attempts):
        return ""
    seq = " -> ".join(f"{a.get('batch')} {'OOM' if a.get('status') == 'FAIL_OOM' else 'OK'}" for a in attempts)
    text = t["ladder"].format(seq=seq)
    if r.get("peak_gb"):
        text += " · " + t["peak"].format(gb=r["peak_gb"])
    return text


def render_one(con: Console, report: dict, verdicts: dict, lang: str, width: int):
    """One complete report box in one language."""
    t = REPORT_TEXT[lang]
    box = BoxWriter(con, width)
    specs = report["specs"]
    inst = report.get("install", {})
    levels = report.get("levels", {})
    con.line("")
    box.top()
    when = (report.get("finished_at") or now_iso()).replace("T", " ")[:16]
    box.row(box.paint("bold", f"LeRobot Doctor {TOOL_VERSION} · {t['title']}") + f"   {when} · {fmt_duration(report.get('seconds', 0))}")
    box.divider()

    # ---- machine ---------------------------------------------------------------------------
    box.heading(t["machine"])
    dev = {"cuda": "CUDA", "mps": "Apple MPS", "cpu": "CPU only"}[specs["accelerator"]]
    if specs.get("nvidia"):
        gpu = specs["nvidia"][0]["name"] + (f"{t['lp']}{t['not_used']}{t['rp']}" if specs["accelerator"] != "cuda" else "")
    elif specs["accelerator"] == "mps":
        gpu = "Apple Silicon"
    elif specs.get("other_gpus"):   # an integrated or AMD GPU: the machine has one, PyTorch does not use it
        gpu = ", ".join(specs["other_gpus"]) + f"{t['lp']}{t['gpu_unused']}{t['rp']}"
    else:
        gpu = "—"
    box.row(f"{specs['os']} · {specs['cpu']} · RAM {specs['ram_gb']} GB", indent=4)
    box.row(f"GPU {gpu} · {dev} {specs.get('device_mem_gb')} GB" + (" · bf16" if specs.get("bf16") else ""), indent=4)
    if inst.get("status") == "PASS":
        box.row(f"{box.mark('ok')} LeRobot {inst.get('lerobot', LEROBOT_VERSION)} · torch {inst.get('torch', '?')} · "
                f"{t['video']} {'torchcodec' if inst.get('torchcodec') else 'pyav'}", indent=4)
    else:
        box.row(f"{box.mark('bad')} {install_failure_text(inst)[0 if lang == 'zh' else 1]}", indent=4)
    box.divider()

    if verdicts.get("basics"):
        box.heading(t["basics"])
        box.row(f"{box.mark(verdicts['basics'].get('mark', 'skip'))} {t['basics_line']}", indent=4)
        for zh, en in verdicts.get("notes", []):
            box.row(f"· {zh if lang == 'zh' else en}", indent=6)
        box.divider()

    if verdicts["levels"]:
        box.heading(t["levels"])
        rows = []
        for lv in LEVELS:
            v = verdicts["levels"][lv.id]
            ri = levels.get(lv.id, {}).get("infer") or {}
            rt = levels.get(lv.id, {}).get("train") or {}
            rows.append([lv.id, lv.label,
                         f"{box.mark(v['infer']['mark'])} {short_infer(v['infer'], ri, t)}",
                         f"{box.mark(v['train']['mark'])} {short_train(v['train'], rt, t)}"])
        box.table(list(t["header"]), rows)
        box.row(box.paint("dim", t["legend"].format(**box.marks)), indent=4)
        box.divider()

        # ---- details: every non-green cell gets its full sentence (the evidence) -------------
        details = []
        for lv in LEVELS:
            v = verdicts["levels"][lv.id]
            ri = levels.get(lv.id, {}).get("infer") or {}
            rt = levels.get(lv.id, {}).get("train") or {}
            for stage, vv in ((t["infer"], v["infer"]), (t["train"], v["train"])):
                if vv["mark"] != "ok" or vv.get("evidence") == "partial":
                    details.append((lv, stage, vv["mark"], vv[lang]))
            ladder = ladder_text(rt, t)
            if ladder:
                details.append((lv, t["train"], "skip", ladder))
            before, after = ri.get("demo") or {}, ri.get("demo_after_training") or {}
            if before.get("mean_abs_err_deg") is not None and after.get("mean_abs_err_deg") is not None:
                details.append((lv, t["train"], "ok", t["after_training"].format(
                    n=after.get("extra_steps", "?"), a=f"{before['mean_abs_err_deg']:.0f}", b=f"{after['mean_abs_err_deg']:.0f}")))
            est = (report.get("estimate") or {}).get(lv.id, {}).get("train_estimate")
            measured = v["train"].get("cloud")
            if est and measured is not None and (est == "cloud") != measured:
                details.append((lv, t["estimate"], "skip", t["est_note"].format(w=t["est_words"][est])))
        if details:
            box.heading(t["details"])
            for lv, stage, mark, text in details:
                box.row(f"{box.mark(mark)} {lv.id} {lv.label} · {stage}{t['sep']}{text}", indent=4)
            box.divider()

    r = verdicts["route"]
    box.heading(f"{t['route']} [{r['rule']}]")
    box.row(box.paint("bold", r[lang]), indent=4)
    box.bottom()


def render_report(con: Console, report: dict, verdicts: dict, path: Path | None = None):
    """The report twice: a complete English box, then a complete Chinese one, then the JSON path."""
    width = max(REPORT_MIN_WIDTH, min(REPORT_MAX_WIDTH, shutil.get_terminal_size((100, 24)).columns - 1))
    for lang in ("en", "zh"):
        render_one(con, report, verdicts, lang, width)
    if path:   # the path on its own line: easy to select, and it never pushes the label past the box width
        con.line(f"  {bi('完整数据（求助时发这个文件）', 'full data (share this file when asking for help)')}:")
        con.line(f"  {path}")


# ----------------------------------------------------------------------------------------------
# Orchestrator (runs inside the venv): dataset, URDF, sim page, ladder, report.
# ----------------------------------------------------------------------------------------------

class Report:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.save()

    def save(self):
        """The dated file and a `report-latest.json` copy, each written atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.data, ensure_ascii=False, indent=2, default=str)   # a stray Path or numpy scalar must not lose the report
        write_atomic(self.path, text)
        write_atomic(self.path.parent / LATEST_REPORT_NAME, text)


def write_atomic(path: Path, text: str):
    """Write next to the target and rename over it: a reader never sees a half-written file, and two
    runs on the same day never interleave into one (the temp name is unique per call)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"   # a character the pipe cannot encode must not end a probe
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Plain HTTP downloads. huggingface_hub's xet transfer depends on a separate CAS CDN that is slow
    # or broken on some networks (measured here: ~1 MB/s and repeated CAS errors vs. full speed
    # over HTTP). A diagnostic tool takes the path that works everywhere.
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT_S))
    env.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_S))
    return env


_PROGRESS_RE = re.compile(r"\d+%\||\d+(\.\d+)?\s?[kMG]?B/s|it/s|Downloading|Fetching")
_STEP_TIME_RE = re.compile(r"(\d+(?:\.\d+)?) s/step")   # the note worker_train puts on every @@progress line


def is_download_progress(text: str) -> bool:
    """tqdm / huggingface_hub progress lines: the only child stdout a beginner should see."""
    return bool(_PROGRESS_RE.search(text))


class WorkerOutput:
    """Parses one worker's @@ lines. Keeps what a run that never sent its @@result still told us:
    the last activity, the progress of every stage and the timed training steps."""

    def __init__(self, con: Console, progress_label: str, on_joints=None):
        self.con, self.label, self.on_joints = con, progress_label, on_joints
        self.result: dict = {}
        self.last_activity = ""
        self.progress: dict = {}
        self.train_stage = None
        self.step_times: list[float] = []
        self.batch = None

    def on_line(self, text: str, is_cr: bool) -> bool:
        if text.startswith("@@result "):
            try:
                self.result.update(json.loads(text[len("@@result "):]))
            except json.JSONDecodeError:
                pass
            return True
        if text.startswith("@@progress "):
            try:
                _, stage, i, n, *extra = text.split(" ", 4)
                i, n = int(i), int(n)
            except ValueError:   # a line the pipe cut half-way: nothing to draw, nothing to keep
                return True
            note = extra[0] if extra else ""
            self.progress[stage] = [i, n, note]
            if stage.startswith("batch"):
                if stage != self.train_stage:   # a smaller batch after an OOM: the earlier steps were another run's
                    self.train_stage, self.step_times = stage, []
                    self.batch = int(stage[5:]) if stage[5:].isdigit() else None
                m = _STEP_TIME_RE.match(note)
                if m and i > TRAIN_WARMUP:      # i is 1-based; the worker times its 0-based steps >= TRAIN_WARMUP
                    self.step_times.append(float(m.group(1)))
            self.con.bar(f"{self.label} {stage}", i, n, note)
            return True
        if text.startswith("@@activity "):
            self.last_activity = text[len("@@activity "):]
            self.con.activity(self.last_activity)
            return True
        if text.startswith("@@note "):
            self.con.line(f"  {self.con.mark('info')} {text[len('@@note '):]}")
            return True
        if text.startswith("@@event "):
            _, name, *payload = text.split(" ", 2)
            body = payload[0] if payload else ""
            if name == "oom":
                self.con.line(f"  {self.con.mark('warn')} {body}")
            elif name == "note":
                self.con.line(f"  {self.con.mark('info')} {body}")
            return True
        if text.startswith("@@joints "):
            if self.on_joints:
                self.on_joints(text[len("@@joints "):])
            return True
        # Everything else from the child goes to the log only, except download progress bars:
        # a traceback or a library warning on a beginner's screen reads as "it broke".
        if is_download_progress(text):
            self.con.transient(text[-200:]) if is_cr else self.con.raw(text + "\n")
        return True

    def partial(self) -> dict:
        return {"phase": self.last_activity, "progress": {k: list(v) for k, v in self.progress.items()},
                "step_times": list(self.step_times), "batch": self.batch}


def run_worker(py: Path, con: Console, log_path: Path, worker_args: list[str], budget_s: int,
               progress_label: str, on_joints=None, env_extra: dict | None = None) -> dict:
    """Run `lerobot_doctor.py --worker ...` in the venv and collect its @@result."""
    out = WorkerOutput(con, progress_label, on_joints)
    cmd = [str(py), str(Path(__file__).resolve()), "--worker", *worker_args]
    t0 = time.monotonic()
    env = child_env()
    env.update(env_extra or {})
    offset = log_path.stat().st_size if log_path.exists() else 0
    rc = stream_process(cmd, con, log_path, env=env, timeout=budget_s, on_line=out.on_line)
    seconds = round(time.monotonic() - t0, 1)
    worker_keys = {k: v for k, v in out.result.items() if k != "status"}
    if rc == -999:
        if out.result.get("status") == "PASS":   # the probe finished and reported; only its bonus work overran
            return {**out.result, "seconds": seconds, "log": str(log_path), "overran_after_result": True}
        return {"status": "TIMEOUT", "evidence": "not_run", "seconds": seconds, "log": str(log_path),
                "partial": out.partial(), **worker_keys}
    if out.result and rc == 0:
        out.result.setdefault("seconds", seconds)
        out.result["log"] = str(log_path)
        return out.result
    tail = log_since(log_path, offset)[-20000:]   # this run's output only, never a previous attempt's
    status = classify_exit(rc, tail)
    if out.result.get("status"):   # the worker classified its own failure before exiting non-zero
        status = out.result["status"]
    err = ""
    for line in reversed(tail.splitlines()):
        if "Error" in line or "error" in line:
            err = line.strip()[:300]
            break
    # The parent's bookkeeping wins over anything the worker wrote under the same key.
    return {**worker_keys, "status": status, "evidence": "measured" if status in STATUS_MEASURED_FAIL else "not_run",
            "returncode": rc, "seconds": seconds, "log": str(log_path), "error": err}


def run_worker_with_download_retry(py, con, log_path, wargs, budget, label, on_joints=None) -> dict:
    """One retry on a download failure: finished files stay in the Hugging Face cache, so a
    second attempt only fetches what is missing."""
    r = run_worker(py, con, log_path, wargs, budget, label, on_joints=on_joints)
    if r.get("status") == "FAIL_DOWNLOAD":
        con.item("warn", "下载中断，重试一次（已下好的部分保留）", "download interrupted; retrying once (finished parts are kept)")
        r = run_worker(py, con, log_path, wargs, budget, label, on_joints=on_joints)
        r["download_retried"] = True
    return r


def orchestrate(args, con: Console, py: Path, report: Report):
    """Dataset, 3D page, the two ladders. Returns the SimPage (or None) so the caller can close it."""
    data = report.data
    specs = data["specs"]
    logs = WORK_DIR / "logs"
    total_steps = 4 + 2 * len(LEVELS)
    step = [3]  # steps 1-3 were specs / confirm / install

    def next_step(zh, en, eta=""):
        step[0] += 1
        con.step(step[0], total_steps, zh, en, eta)

    # ---- dataset --------------------------------------------------------------------------
    next_step("下载样例数据集（官方 SO-101 抓取放置录像，前 5 集）", f"downloading the sample dataset ({DATASET_REPO}, first 5 episodes)", "几十 MB / tens of MB")
    con.activity(bi("下载样例数据", "downloading sample data"))
    ds = run_worker(py, con, logs / "dataset.log", ["dataset"], 15 * 60, "dataset")
    data["dataset"] = ds
    report.save()
    if ds.get("status") != "PASS":
        con.item("bad", f"样例数据没准备好：{ds.get('error') or ds.get('status')}", f"sample dataset not ready: {ds.get('error') or ds.get('status')}")
        return None
    con.item("ok", f"{ds['frames']} 帧 · 指令「{ds['task']}」· 解码后端 {ds['video_backend']}",
             f"{ds['frames']} frames, task '{ds['task']}', decoder {ds['video_backend']}")

    # ---- sim page -----------------------------------------------------------------------------
    # The page is a bonus: it never blocks a probe, and any exception it raises later closes the
    # page and lets the probes continue.
    page = {"sim": None}
    if not args.no_sim:
        con.activity(bi("准备 3D 模拟页面", "preparing the 3D simulation page"))
        try:
            page["sim"] = SimPage(port=args.port, urdf_dir=WORK_DIR / "models" / "so101", dataset_root=WORK_DIR / "dataset",
                                  con=con, open_browser=not args.no_browser)
            url = f"http://127.0.0.1:{page['sim'].port}"
            con.item("ok", f"3D 模拟页面：{url}（浏览器应已自动打开）", f"3D simulation page: {url} (a browser tab should have opened)")
        except Exception as e:  # noqa: BLE001
            busy = "address already in use" in str(e).lower() or "errno 98" in str(e).lower() or "10048" in str(e)
            hint_zh = f"；端口 {args.port} 被占，可加 --port 换一个" if busy else ""
            hint_en = f"; port {args.port} is busy, pick another with --port" if busy else ""
            con.item("warn", f"3D 页面没起来（{type(e).__name__}: {str(e)[:120]}）{hint_zh}，探针照跑",
                     f"3D page failed ({type(e).__name__}: {str(e)[:120]}){hint_en}; probes continue")

    def sim_call(method: str, *a):
        s = page["sim"]
        if not s:
            return
        try:
            getattr(s, method)(*a)
        except Exception as e:  # noqa: BLE001 - a NaN action, a dropped websocket, a renamed joint
            page["sim"] = None
            con.item("warn", f"3D 页面出错（{type(e).__name__}: {str(e)[:80]}），已关闭页面，探针照跑",
                     f"3D page failed ({type(e).__name__}: {str(e)[:80]}); page closed, probes continue")
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- weights ------------------------------------------------------------------------------
    con.activity(bi("查询模型大小", "querying model sizes"))
    weights = {}
    for lv in LEVELS:
        if lv.pretrained:
            weights[lv.id] = hf_model_summary(lv.weights) or {}
    data["weights"] = weights
    dtype = "bfloat16" if specs["bf16"] else "float32"
    device = args.device or specs["accelerator"]
    total_dl = sum((w.get("bytes") or 0) for lid, w in weights.items())
    con.item("info", f"权重总计约 {gb(total_dl)} GB（已缓存的不重下）· 设备 {device} · 大模型精度 {dtype}",
             f"weights total ~{gb(total_dl)} GB (cached files are not fetched again), device {device}, large-model dtype {dtype}")
    report.save()

    results = data.setdefault("levels", {})
    selected = {x.strip().upper() for x in args.levels.split(",")} if args.levels else {lv.id for lv in LEVELS}
    filtered = {"status": "NOT_RUN", "evidence": "not_run", "reason": "user_filtered",
                "zh": "未测：这次运行没有选它（--levels）", "en": "not tested: not selected for this run (--levels)"}

    def joints_cb(payload):
        sim_call("on_joints", payload)

    # ---- inference ladder -------------------------------------------------------------------
    for lv in LEVELS:
        next_step(f"{lv.label} 推理 + 模拟执行", f"{lv.label} inference + simulated task", "1–10 min")
        results.setdefault(lv.id, {})
        pre = None if lv.id in selected else dict(filtered)
        pre = pre or infer_precheck(lv, results, specs, weights, dtype)
        if pre:
            results[lv.id]["infer"] = pre
            con.item("skip" if pre["evidence"] == "not_run" else "bad", pre["zh"], pre["en"])
            report.save()
            continue
        sim_call("begin_level", lv.label)
        con.activity(f"{lv.label} · " + bi("下载/加载模型", "downloading/loading model"))
        wargs = ["infer", "--level", lv.id, "--device", device, "--dtype", dtype]
        if args.vram_cap:
            wargs += ["--vram-cap", str(args.vram_cap)]
        r = run_worker_with_download_retry(py, con, logs / f"{lv.id}-infer.log", wargs, INFER_BUDGET_S, lv.label, on_joints=joints_cb)
        r["dtype"] = dtype
        r["device"] = device
        r["params"] = (weights.get(lv.id) or {}).get("params") or r.get("params")
        results[lv.id]["infer"] = r
        v = infer_verdict(lv, r)
        con.item(v["mark"], v["zh"], v["en"])
        sim_call("end_level", lv.label, r)
        report.save()

    # ---- training ladder --------------------------------------------------------------------
    for lv in LEVELS:
        next_step(f"{lv.label} 训练", f"{lv.label} training", "2–15 min")
        pre = None if lv.id in selected else dict(filtered)
        pre = pre or train_precheck(lv, results)
        if pre:
            results[lv.id]["train"] = pre
            con.item(train_verdict(lv, pre)["mark"], pre["zh"], pre["en"])
            report.save()
            continue
        con.activity(f"{lv.label} · " + bi("加载模型准备训练", "loading model for training"))
        wargs = ["train", "--level", lv.id, "--device", device, "--dtype", dtype]
        if args.vram_cap:
            wargs += ["--vram-cap", str(args.vram_cap)]
        budget = TRAIN_L1_BUDGET_S if lv.id == "L1" else TRAIN_BUDGET_S
        if lv.id == "L1":
            sim_call("begin_level", lv.label + " " + bi("（训练后）", "(after training)"))
        r = run_worker_with_download_retry(py, con, logs / f"{lv.id}-train.log", wargs, budget, lv.label, on_joints=joints_cb)
        r["device"] = device   # train_precheck's speed rule reads it off the smaller level's result
        r["params"] = (weights.get(lv.id) or {}).get("params") or r.get("params")
        results[lv.id]["train"] = r
        if r.get("status") == "PASS":
            r["hours"] = round(projected_hours(r["update_s"], r["batch"]), 1)
        elif r.get("status") == "TIMEOUT" and (pn := partial_train_numbers(r)):
            # the probe hit its budget, but every timed step it finished is a measurement
            r.update(update_s=pn["update_s"], batch=pn["batch"], hours=pn["hours"], timed_steps=pn["n"])
        v = train_verdict(lv, r)
        con.item(v["mark"], v["zh"], v["en"])
        if lv.id == "L1" and r.get("demo"):
            results[lv.id]["infer"]["demo_after_training"] = r["demo"]
        report.save()

    sim_call("finish")
    return page["sim"]


# ----------------------------------------------------------------------------------------------
# 3D simulation page (viser). Lives in the orchestrator; the worker only sends joint targets.
# ----------------------------------------------------------------------------------------------

def ensure_urdf(urdf_dir: Path, con: Console | None = None) -> Path:
    urdf_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{URDF_REPO}/{URDF_COMMIT}/"
    files = [(URDF_FILE, "so101_new_calib.urdf"), ("LICENSE", "LICENSE")] + \
            [(f"Simulation/SO101/assets/{m}", f"assets/{m}") for m in URDF_MESHES]
    for remote, local in files:
        dst = urdf_dir / local
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if con:
            con.activity(bi(f"下载 3D 模型 {local}", f"downloading 3D model {local}"))
        # .part then rename: a Ctrl-C or a dropped connection must not leave a truncated mesh that
        # the `dst.exists()` check above would then keep forever.
        part = dst.with_name(dst.name + ".part")
        with urllib.request.urlopen(base + remote, timeout=60) as resp:
            part.write_bytes(resp.read())
        os.replace(part, dst)
    return urdf_dir / "so101_new_calib.urdf"


class SimPage:
    """The browser page: SO-101 URDF driven by the worker's joint targets, plus the camera frames."""

    def __init__(self, port: int, urdf_dir: Path, dataset_root: Path, con: Console, open_browser=True):
        import numpy as np
        import viser
        from viser.extras import ViserUrdf

        self.np = np
        self.con = con
        urdf_path = ensure_urdf(urdf_dir, con)
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="LeRobot Doctor · SO-101", verbose=False)
        get_port = getattr(self.server, "get_port", None)   # viser may have moved to a free port; print the real one
        self.port = int(get_port()) if callable(get_port) else port
        scene = self.server.scene
        scene.set_up_direction("+z")
        scene.add_grid("/ground", width=0.8, height=0.8, plane="xy", cell_size=0.05)
        self.arm = ViserUrdf(self.server, urdf_path, root_node_name="/so101")

        @self.server.on_client_connect
        def _(client) -> None:   # the arm is 30 cm tall; viser's default camera sits metres away
            client.camera.position = (0.55, -0.55, 0.40)
            client.camera.look_at = (0.0, 0.0, 0.12)

        self.joint_names = list(self.arm.get_actuated_joint_names())
        self.limits = {name: (float(lo if lo is not None else -math.pi), float(hi if hi is not None else math.pi))
                       for name, (lo, hi) in self.arm.get_actuated_joint_limits().items()}
        gui = self.server.gui
        with gui.add_folder(bi("当前", "Now")):
            self.level_text = gui.add_text(bi("级别", "level"), initial_value="—", disabled=True)
            self.task_text = gui.add_text(bi("指令", "task"), initial_value="", disabled=True)
            self.hz_text = gui.add_text(bi("控制频率", "control Hz"), initial_value="—", disabled=True)
            self.lat_text = gui.add_text(bi("每块延迟", "chunk latency"), initial_value="—", disabled=True)
            self.status_md = gui.add_markdown("")
        with gui.add_folder(bi("样例相机", "Sample cameras")):
            blank = np.zeros((120, 160, 3), dtype=np.uint8)
            self.img_up = gui.add_image(blank, label="up")
            self.img_side = gui.add_image(blank, label="side")
        with gui.add_folder(bi("关节 (度)：模型 / 真人", "Joints (deg): model / human")):
            self.plots = {}
            for name in ARM_JOINTS + (GRIPPER_JOINT,):
                self.plots[name] = gui.add_text(name, initial_value="—", disabled=True)
        self._frames = None
        self._dataset_root = dataset_root
        self._frame_idx = 0
        self._last_img = -1
        self.home()
        if open_browser:
            import webbrowser
            url = f"http://127.0.0.1:{self.port}"
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    def home(self):
        self.arm.update_cfg(self.np.zeros(len(self.joint_names)))

    def _load_frames(self):
        """Episode 0 camera frames, decoded once, downscaled for the page."""
        if self._frames is not None:
            return
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(DATASET_REPO, root=self._dataset_root, episodes=[DEMO_EPISODE])
            cams = ds.meta.camera_keys
            frames = []
            for i in range(min(len(ds), DEMO_SECONDS * CONTROL_FPS)):
                item = ds[i]
                pair = []
                for key in cams[:2]:
                    img = item[key]
                    if img.dtype != self.np.uint8:
                        img = (img * 255).clamp(0, 255).to("cpu").numpy().astype(self.np.uint8)
                    else:
                        img = img.numpy()
                    img = self.np.transpose(img, (1, 2, 0))[::4, ::4]   # 480x640 -> 120x160
                    pair.append(self.np.ascontiguousarray(img))
                frames.append((pair, item["action"].numpy()))
            self._frames = frames
            self.task_text.value = ds.meta.tasks.index[0] if hasattr(ds.meta.tasks, "index") else ""
        except Exception as e:  # noqa: BLE001
            self._frames = []
            self.con.line(f"  (sim page: could not load frames: {type(e).__name__}: {str(e)[:100]})")

    def begin_level(self, label: str):
        self.level_text.value = label
        self.status_md.content = bi("加载模型中…", "loading the model…")
        self.hz_text.value = "—"
        self.lat_text.value = "—"
        self._frame_idx = 0
        self._last_img = -1
        for name in ARM_JOINTS + (GRIPPER_JOINT,):
            self.plots[name].value = "—"
        self.home()
        threading.Thread(target=self._load_frames, daemon=True).start()

    def on_joints(self, payload: str):
        """'t q1 q2 q3 q4 q5 g hz lat_ms' from the worker: degrees for arm joints, percent for the gripper."""
        parts = payload.split()
        if len(parts) < 7:
            return
        t = int(parts[0])
        q = [float(x) for x in parts[1:7]]
        cfg = []
        for name in self.joint_names:
            if name == GRIPPER_JOINT:
                lo, hi = self.limits[name]
                cfg.append(lo + (hi - lo) * min(max(q[5], 0.0), 100.0) / 100.0)
            elif name in ARM_JOINTS:
                cfg.append(math.radians(q[ARM_JOINTS.index(name)]))
            else:
                cfg.append(0.0)
        self.arm.update_cfg(self.np.array(cfg))
        if len(parts) >= 9:
            self.hz_text.value = f"{float(parts[7]):.0f} Hz"
            self.lat_text.value = f"{float(parts[8]):.0f} ms"
        self.status_md.content = bi(f"模拟执行中 · 第 {t} / {DEMO_SECONDS * CONTROL_FPS} 步", f"running the simulated task, step {t} / {DEMO_SECONDS * CONTROL_FPS}")
        if self._frames and t < len(self._frames) and t - self._last_img >= 3:
            (up, side), human = self._frames[t]
            self.img_up.image = up
            self.img_side.image = side
            self._last_img = t
            for i, name in enumerate(ARM_JOINTS + (GRIPPER_JOINT,)):
                self.plots[name].value = f"model {q[i]:7.1f}   human {float(human[i]):7.1f}"

    def end_level(self, label: str, result: dict):
        st = result.get("status")
        self.status_md.content = f"**{label}**: {st}" + (f" · {result.get('latency_ms', 0):.0f} ms/chunk" if result.get("latency_ms") else "")

    def finish(self):
        self.status_md.content = bi("体检完成，报告在终端。", "Check finished; the report is in the terminal.")

    def close(self):
        """Stop the server: the process must not depend on viser's threads happening to be daemonic."""
        stop = getattr(self.server, "stop", None)
        if callable(stop):
            stop()


# ----------------------------------------------------------------------------------------------
# Worker (child process inside the venv): dataset / infer / train.
# ----------------------------------------------------------------------------------------------

def emit(kind: str, *parts):
    print(f"@@{kind} " + " ".join(str(p) for p in parts), flush=True)


def emit_result(d: dict):
    print("@@result " + json.dumps(d, ensure_ascii=False), flush=True)


def worker_dataset() -> None:
    import torch  # noqa: F401 - forces the heavy import here so the parent sees the crash in this log
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.import_utils import get_safe_default_video_backend

    emit("activity", bi("下载样例数据", "downloading sample data"))
    root = WORK_DIR / "dataset"
    ds = LeRobotDataset(DATASET_REPO, root=root, episodes=DATASET_EPISODES)
    emit("activity", bi("解码第一帧，检查视频链路", "decoding the first frame to test the video path"))
    item = ds[0]
    task = ""
    try:
        task = str(ds.meta.tasks.index[0])
    except Exception:  # noqa: BLE001
        task = str(item.get("task", ""))
    emit_result({"status": "PASS", "frames": ds.num_frames, "episodes": ds.num_episodes,
                 "cameras": list(ds.meta.camera_keys), "fps": ds.fps, "task": task,
                 "video_backend": get_safe_default_video_backend(), "root": str(root)})


def _torch_device(name: str):
    import torch
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(name)


def _apply_vram_cap(device, cap_gb: float | None):
    import torch
    if cap_gb and device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        frac = min(1.0, cap_gb * 1e9 / total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        emit("event", "note", f"simulating a {cap_gb} GB GPU (memory fraction {frac:.2f})")


def _peak_mem_gb(device) -> float | None:
    import torch
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS, kilobytes on Linux
        return round(rss / (1e9 if platform.system() == "Darwin" else 1e6), 2)
    except ImportError:   # Windows has no `resource`; RSS is not reported there
        return None


def _is_oom(exc: BaseException) -> bool:
    import torch
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def _free_cache(device):
    import gc
    import torch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_level_config(level: Level, device, dtype: str, meta, for_training: bool, checkpoint: Path | None = None):
    """The policy config the way lerobot-train / lerobot-eval would resolve it, plus the camera rename map."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.policies.factory import make_policy_config

    rename_map = {}
    src = str(checkpoint) if checkpoint else level.source
    if src:
        cfg = PreTrainedConfig.from_pretrained(src)
        cfg.pretrained_path = src
        cfg.device = str(device)
        if hasattr(cfg, "dtype") and level.large:
            cfg.dtype = dtype
        if for_training and hasattr(cfg, "gradient_checkpointing") and level.large:
            cfg.gradient_checkpointing = True
        ckpt_cams = [k for k, f in cfg.input_features.items() if f.type is FeatureType.VISUAL]
        ds_cams = list(meta.camera_keys)
        if ckpt_cams and set(ckpt_cams) != set(ds_cams) and not checkpoint:
            # The checkpoint was trained with other camera names; lerobot's own answer is --rename_map.
            rename_map = dict(zip(ds_cams, ckpt_cams))
            emit("event", "note", f"camera rename for {level.label}: {rename_map}")
    else:
        cfg = make_policy_config(level.policy, device=str(device))
        if hasattr(cfg, "dtype") and level.large:
            cfg.dtype = dtype
        if for_training and hasattr(cfg, "gradient_checkpointing") and level.large:
            cfg.gradient_checkpointing = True
    return cfg, rename_map


def build_policy(cfg, rename_map: dict, meta, device, for_training: bool):
    """Policy + pre/post processors from a resolved config (mirrors lerobot_train.py / lerobot_eval.py)."""
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy = make_policy(cfg, ds_meta=meta, rename_map=rename_map or None)
    policy.eval()
    kwargs = {}
    if cfg.pretrained_path:
        kwargs["pretrained_path"] = str(cfg.pretrained_path)
        kwargs["preprocessor_overrides"] = {"device_processor": {"device": str(device)},
                                            "rename_observations_processor": {"rename_map": rename_map}}
        if for_training:
            kwargs["dataset_stats"] = meta.stats
            kwargs["preprocessor_overrides"]["normalizer_processor"] = {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping, "stats": meta.stats}
            kwargs["postprocessor_overrides"] = {"unnormalizer_processor": {
                "features": policy.config.output_features, "norm_map": policy.config.normalization_mapping,
                "stats": meta.stats}}
    else:
        kwargs["dataset_stats"] = meta.stats
    pre, post = make_pre_post_processors(policy_cfg=cfg, **kwargs)
    return policy, pre, post


def load_level_policy(level: Level, device, dtype: str, meta, for_training: bool, checkpoint: Path | None = None):
    cfg, rename_map = build_level_config(level, device, dtype, meta, for_training, checkpoint)
    policy, pre, post = build_policy(cfg, rename_map, meta, device, for_training)
    return policy, pre, post, rename_map


def training_dataset(cfg, meta):
    """The dataset lerobot-train would build for this policy: action chunks / observation stacks
    come from the policy's delta indices."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta = resolve_delta_timestamps(cfg, meta)
    return LeRobotDataset(DATASET_REPO, root=WORK_DIR / "dataset", episodes=DATASET_EPISODES, delta_timestamps=delta)


def _frame(item, camera_keys) -> dict:
    """The camera tensors of one dataset item as uint8: a decoded frame kept in memory then costs
    0.9 MB instead of 3.7 MB, so a whole demo episode (300 frames x 2 cameras) fits in ~0.6 GB.
    lerobot decodes video to uint8 and divides by 255; going back is exact."""
    import torch
    out = {}
    for key in camera_keys:
        img = item[key]
        if img.dtype != torch.uint8:
            img = (img * 255).round().clamp(0, 255).to(torch.uint8)
        out[key] = img
    return out


def _observation(frame: dict, state, device, task: str, robot_type: str):
    """One inference observation: an already decoded frame + the simulated arm's joint state."""
    import torch
    obs = {key: (img.to(torch.float32) / 255.0).unsqueeze(0).to(device) for key, img in frame.items()}
    obs["observation.state"] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    obs["task"] = task
    obs["robot_type"] = robot_type
    return obs


def worker_infer(level: Level, device_name: str, dtype: str, vram_cap: float | None) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = _torch_device(device_name)
    _apply_vram_cap(device, vram_cap)
    root = WORK_DIR / "dataset"
    ds = LeRobotDataset(DATASET_REPO, root=root, episodes=[DEMO_EPISODE])
    cams = list(ds.meta.camera_keys)
    try:
        task = str(ds.meta.tasks.index[0])
    except Exception:  # noqa: BLE001 - same guard as worker_dataset
        task = str(ds[0].get("task", ""))
    robot_type = ds.meta.robot_type or "so101_follower"

    emit("activity", f"{level.label} · " + bi(f"下载/加载模型到 {device_name}", f"downloading/loading model to {device_name}"))
    t_load = time.monotonic()
    try:
        policy, pre, post, _ = load_level_policy(level, device, dtype, ds.meta, for_training=False)
    except BaseException as e:  # noqa: BLE001 - classified below
        if _is_oom(e):
            emit_result({"status": "FAIL_OOM", "evidence": "measured", "phase": "load", "peak_gb": _peak_mem_gb(device)})
            return {"status": "FAIL_OOM"}
        raise
    load_s = round(time.monotonic() - t_load, 1)
    params = sum(p.numel() for p in policy.parameters())
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    result = {"status": "PASS", "evidence": "measured", "params": params, "load_s": load_s,
              "n_action_steps": n_action_steps, "device": device_name, "dtype": dtype,
              "timing_mode": "reset_per_chunk"}   # every timed call is one full chunk prediction, never an amortised pop

    state0 = ds[0]["observation.state"].numpy().tolist()
    # ---- timing: predict_action_chunk is the real forward pass --------------------------
    is_gpu = device.type == "cuda"
    warm, timed = (WARMUP_GPU, TIMED_GPU) if is_gpu else (WARMUP_CPU, TIMED_CPU)
    budget = n_action_steps / CONTROL_FPS
    lat = []
    emit("activity", f"{level.label} · " + bi("预热", "warm-up"))
    try:
        with torch.inference_mode():
            # select_action on an empty queue = exactly one forward pass (predict_action_chunk)
            # plus a queue pop. reset() before each call keeps every call a real forward, and this
            # is the same code path lerobot-record uses on the robot.
            for i in range(warm):
                obs = _observation(_frame(ds[i], cams), state0, device, task, robot_type)
                policy.reset()
                policy.select_action(pre(obs))
                _sync(device)
                emit("progress", "warmup", i + 1, warm)
            for i in range(timed):
                obs = _observation(_frame(ds[i], cams), state0, device, task, robot_type)   # decoded before the clock starts
                policy.reset()
                t0 = time.perf_counter()
                policy.select_action(pre(obs))
                _sync(device)
                dt = time.perf_counter() - t0
                lat.append(dt)
                emit("progress", "timing", i + 1, timed, f"median {statistics.median(lat)*1000:.0f} ms")
                if i == 0 and not is_gpu and dt > HOPELESS_RATIO * budget:
                    emit("event", "note", f"one forward pass already {dt/budget:.0f}x over the {budget:.2f} s budget; stopping the timing here")
                    break
    except BaseException as e:  # noqa: BLE001
        if _is_oom(e):
            emit_result({**result, "status": "FAIL_OOM", "phase": "forward", "peak_gb": _peak_mem_gb(device)})
            return {"status": "FAIL_OOM"}
        raise
    result.update(**infer_summary(lat, n_action_steps), peak_gb=_peak_mem_gb(device))
    # ---- simulated task ------------------------------------------------------------------
    if result["status"] in STATUS_MEASURED_OK:
        result["demo"] = run_demo(policy, pre, post, ds, device, task, robot_type, state0, n_action_steps)
    emit_result(result)
    return result


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_demo(policy, pre, post, ds, device, task, robot_type, state0, n_action_steps: int) -> dict:
    """Half-closed loop: recorded frames in, model actions drive a kinematic arm whose joint
    state is fed back as observation.state. Joint targets stream to the parent as @@joints.

    Every frame is decoded before the clock starts: a real robot's cameras hand over frames for
    free, so the decoder's time (seconds per frame with pyav on a slow CPU) is not the policy's.
    time.perf_counter throughout: time.monotonic ticks every 15.6 ms on Windows, half a 30 Hz period."""
    import torch
    n_avail = min(len(ds), DEMO_SECONDS * CONTROL_FPS)
    cams = list(ds.meta.camera_keys)
    emit("activity", bi(f"解码 {n_avail} 帧样例画面", f"decoding {n_avail} sample frames"))
    t_dec = time.perf_counter()
    frames, human = [], []
    for t in range(n_avail):
        item = ds[t]
        frames.append(_frame(item, cams))
        human.append(item["action"].numpy().tolist()[:6])
        if t % 50 == 0:
            emit("progress", "decode", t + 1, n_avail)
    decode_s = time.perf_counter() - t_dec
    sim = list(state0)
    policy.reset()
    errs, refill = [], []
    done = 0
    emit("activity", bi("模拟执行样例任务", "running the simulated task"))
    t_start = time.perf_counter()
    with torch.inference_mode():
        for t in range(n_avail):
            tick = time.perf_counter()
            obs = _observation(frames[t], sim, device, task, robot_type)
            t0 = time.perf_counter()
            action = post(policy.select_action(pre(obs)))
            _sync(device)
            dt = time.perf_counter() - t0
            if t % n_action_steps == 0:   # the queue is empty here: a real forward pass, not a pop
                refill.append(dt)
            target = action[0].detach().to("cpu").float().numpy().tolist()[:6]
            sim = [s + DEMO_FOLLOW_ALPHA * (tg - s) for s, tg in zip(sim, target)]
            errs.append(sum(abs(a - b) for a, b in zip(target[:5], human[t][:5])) / 5)
            done = t + 1
            elapsed = time.perf_counter() - t_start
            hz = done / elapsed if elapsed > 0 else 0.0
            emit("joints", t, *[f"{x:.3f}" for x in sim], f"{hz:.1f}", f"{(refill[-1] if refill else 0.0) * 1000:.1f}")
            if t % 15 == 0:
                emit("progress", "demo", done, n_avail, f"{hz:.0f} Hz")
            if elapsed > DEMO_WALL_CAP_S:
                break
            sleep_for = 1.0 / CONTROL_FPS - (time.perf_counter() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
    out = demo_summary(done, n_avail, time.perf_counter() - t_start, refill)
    out.update(decode_s=round(decode_s, 1), mean_abs_err_deg=round(statistics.mean(errs), 2) if errs else None)
    return out


def worker_train(level: Level, device_name: str, dtype: str, vram_cap: float | None) -> dict:
    import torch
    from accelerate import Accelerator
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    device = _torch_device(device_name)
    _apply_vram_cap(device, vram_cap)
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(DATASET_REPO, root=WORK_DIR / "dataset")
    cfg_policy, rename_map = build_level_config(level, device, dtype, meta, for_training=True)
    ds = training_dataset(cfg_policy, meta)
    result = {"status": "NOT_RUN", "evidence": "measured", "device": device_name, "dtype": dtype, "attempts": []}
    accelerator = Accelerator(step_scheduler_with_optimizer=False, cpu=(device.type == "cpu"),
                              mixed_precision="no")
    policy = None
    for batch_size in level.batches:
        _free_cache(device)
        emit("activity", f"{level.label} · batch {batch_size} · " + bi("加载模型", "loading model"))
        try:
            if policy is None:
                policy, pre, post = build_policy(cfg_policy, rename_map, meta, device, for_training=True)
            policy.train()
            params = sum(p.numel() for p in policy.parameters())
            result["params"] = params
            # make_optimizer_and_scheduler reads four fields of TrainPipelineConfig; the policy's own
            # presets are what `lerobot-train` uses by default (use_policy_training_preset=True).
            cfg = SimpleNamespace(use_policy_training_preset=True, optimizer=policy.config.get_optimizer_preset(),
                                  scheduler=policy.config.get_scheduler_preset(), steps=10_000)
            optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=device.type == "cuda", drop_last=True)
            it = iter(loader)
            meters = {"loss": AverageMeter("loss", ":.3f"), "grad_norm": AverageMeter("grdn", ":.3f"),
                      "lr": AverageMeter("lr", ":0.1e"), "update_s": AverageMeter("updt_s", ":.3f"),
                      "dataloading_s": AverageMeter("data_s", ":.3f")}
            if torch.cuda.is_available():   # update_policy writes this meter whenever CUDA exists
                meters["gpu_mem_gb"] = AverageMeter("mem_gb", ":.2f")
            tracker = MetricsTracker(batch_size, ds.num_frames, ds.num_episodes, meters, initial_step=0)
            times = []
            n_total = TRAIN_WARMUP + TRAIN_TIMED
            for i in range(n_total):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                for cam in ds.meta.camera_keys:
                    if cam in batch and batch[cam].dtype == torch.uint8:
                        batch[cam] = batch[cam].to(torch.float32) / 255.0
                batch = pre(batch)
                t0 = time.perf_counter()
                tracker, _ = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm,
                                           accelerator=accelerator, lr_scheduler=lr_scheduler)
                _sync(device)
                dt = time.perf_counter() - t0
                if i >= TRAIN_WARMUP:
                    times.append(dt)
                emit("progress", f"batch{batch_size}", i + 1, n_total,
                     f"{dt:.2f} s/step" + (f" peak {_peak_mem_gb(device)} GB" if device.type != 'cpu' else ""))
            med = statistics.median(times)
            attempt = {"batch": batch_size, "update_s": round(med, 3), "peak_gb": _peak_mem_gb(device),
                       "loss": round(float(tracker.metrics["loss"].avg), 4)}
            result["attempts"].append(attempt)
            result.update(status="PASS", batch=batch_size, update_s=attempt["update_s"], peak_gb=attempt["peak_gb"])
            break
        except BaseException as e:  # noqa: BLE001
            if _is_oom(e):
                emit("event", "oom", f"{level.label} batch {batch_size}: out of memory, trying a smaller batch")
                result["attempts"].append({"batch": batch_size, "status": "FAIL_OOM"})
                # Drop every reference of the failed attempt (the optimizer alone holds 2x the weights)
                # before asking the allocator for room, or the smaller batch OOMs on the leftovers.
                policy = optimizer = lr_scheduler = loader = it = batch = tracker = None
                _free_cache(device)
                cfg_policy, rename_map = build_level_config(level, device, dtype, meta, for_training=True)
                continue
            raise
    if result["status"] != "PASS":
        result.update(status="FAIL_OOM", evidence="measured")
        emit_result(result)
        return result
    emit_result(result)   # the measurement is on record now; the bonus demo below may overrun the budget
    if level.id == "L1":
        result["demo"] = _train_more_and_demo(policy, pre, post, ds, device, optimizer, lr_scheduler, accelerator,
                                             tracker, cfg, result["batch"], result["update_s"])
        emit_result(result)
    return result


def extra_training_steps(update_s: float) -> int:
    """Pure. How many extra ACT steps fit the demo budget: 300 on a GPU, fewer on a slow CPU."""
    if update_s <= 0:
        return ACT_DEMO_STEPS
    return max(1, min(ACT_DEMO_STEPS, int(ACT_DEMO_BUDGET_S / update_s)))


def _train_more_and_demo(policy, pre, post, ds, device, optimizer, lr_scheduler, accelerator, tracker, cfg, batch_size,
                         update_s: float):
    """Keep training ACT (up to ACT_DEMO_STEPS, within ACT_DEMO_BUDGET_S), then run the simulated task."""
    import torch
    from lerobot.scripts.lerobot_train import update_policy

    n_extra = extra_training_steps(update_s)
    if n_extra < ACT_DEMO_MIN_STEPS:   # a handful of steps changes nothing visible; say so instead of pretending
        emit("event", "note", f"only {n_extra} extra ACT steps fit the {ACT_DEMO_BUDGET_S} s demo budget at {update_s:.1f} s/step; skipping the after-training demo")
        return {"status": "SKIPPED", "reason": "too_few_steps", "extra_steps": n_extra,
                "trained_steps": TRAIN_WARMUP + TRAIN_TIMED}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)
    emit("activity", bi(f"ACT 继续训练 {n_extra} 步", f"training ACT for {n_extra} more steps"))
    for step in range(n_extra):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        for cam in ds.meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].to(torch.float32) / 255.0
        batch = pre(batch)
        tracker, _ = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm,
                                   accelerator=accelerator, lr_scheduler=lr_scheduler)
        if step % 10 == 0:
            emit("progress", "train", step + 1, n_extra, f"loss {float(tracker.metrics['loss'].avg):.3f}")
    ckpt = WORK_DIR / "checkpoints" / "act"
    try:
        policy.save_pretrained(ckpt)
        pre.save_pretrained(ckpt)
        post.save_pretrained(ckpt)
    except Exception as e:  # noqa: BLE001 - the demo does not need the files on disk
        emit("event", "note", f"checkpoint not saved: {type(e).__name__}")
    policy.eval()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    demo_ds = LeRobotDataset(DATASET_REPO, root=WORK_DIR / "dataset", episodes=[DEMO_EPISODE])   # single frames, no chunks
    try:
        task = str(ds.meta.tasks.index[0])
    except Exception:  # noqa: BLE001
        task = str(demo_ds[0].get("task", ""))
    state0 = demo_ds[0]["observation.state"].numpy().tolist()
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    demo = run_demo(policy, pre, post, demo_ds, device, task, ds.meta.robot_type or "so101_follower", state0, n_action_steps)
    demo["extra_steps"] = n_extra
    demo["trained_steps"] = n_extra + TRAIN_WARMUP + TRAIN_TIMED
    return demo


def worker_main(argv: list[str]) -> int:
    _make_streams_non_fatal()   # the parent reads a pipe; a character it cannot encode must not end the probe
    p = argparse.ArgumentParser(prog="lerobot_doctor --worker")
    p.add_argument("kind", choices=["dataset", "infer", "train"])
    p.add_argument("--level")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--vram-cap", type=float, default=None)
    a = p.parse_args(argv)
    if a.kind == "dataset":
        worker_dataset()
        return 0
    level = LEVEL_BY_ID[a.level]
    if a.kind == "infer":
        worker_infer(level, a.device, a.dtype, a.vram_cap)
    else:
        worker_train(level, a.device, a.dtype, a.vram_cap)
    return 0


# ----------------------------------------------------------------------------------------------
# Crash handling. The tool's own failure must never end as a blank, closed window: write a log with
# everything a bug report needs, say where it is, and on Windows wait for Enter when nobody else will.
# ----------------------------------------------------------------------------------------------

CRASH_ENV_KEYS = ("PYTHONIOENCODING", "PYTHONUTF8", "DOCTOR_LAUNCHER", "DOCTOR_TAG", "HF_ENDPOINT", "UV_HTTP_TIMEOUT")


def _make_streams_non_fatal():
    """A character the console cannot encode becomes '?' instead of a UnicodeEncodeError that ends the run.
    On Windows, a console stuck in a legacy code page (a bare `python lerobot_doctor.py` without the
    launcher's PYTHONIOENCODING) is switched to UTF-8 so the Chinese half is not printed as '?'."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not hasattr(stream, "reconfigure"):
                continue
            enc = getattr(stream, "encoding", None) or "ascii"
            try:
                "体检 ✓".encode(enc)
                stream.reconfigure(errors="replace")
            except (UnicodeEncodeError, LookupError):
                if platform.system() == "Windows":
                    stream.reconfigure(encoding="utf-8", errors="replace")
                else:
                    stream.reconfigure(errors="replace")
        except (ValueError, OSError):   # closed or exotic stream: leave it alone
            pass


def new_report_path() -> Path:
    """One file per run: two runs on the same day (a failure, then a rerun) must both survive."""
    return WORK_DIR / f"report-{_dt.datetime.now().strftime('%Y-%m-%d-%H%M')}.json"


CURRENT_RUN = {"report": None}   # the report path of this process's run, for the Ctrl-C and crash messages


def crash_location(exc: BaseException) -> str:
    """`lerobot_doctor.py:123 in collect_specs`: the innermost frame of this file, else of anything."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "?"
    ours = [f for f in frames if Path(f.filename).name == Path(__file__).name]
    f = (ours or frames)[-1]
    return f"{Path(f.filename).name}:{f.lineno} in {f.name}"


def _log_listing(logs: Path) -> str:
    try:
        return ", ".join(f"{p.name} ({p.stat().st_size} B)" for p in sorted(logs.glob("*.log"))) or "(none)"
    except OSError:
        return "(unreadable)"


def write_crash_log(exc: BaseException, role: str, report_path: Path | None) -> Path:
    """Everything a bug report needs, in one file. Falls back to the temp dir if WORK_DIR is not writable."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs = WORK_DIR / "logs"
    text = "\n".join([
        f"LeRobot Doctor {TOOL_VERSION} crash report",
        f"time: {now_iso()}",
        f"role: {role}",
        f"platform: {platform.platform()} ({platform.machine()})",
        f"python: {sys.version.split()[0]} at {sys.executable}",
        f"argv: {sys.argv}",
        f"cwd: {os.getcwd()}",
        f"stdout: encoding={getattr(sys.stdout, 'encoding', None)} tty={bool(getattr(sys.stdout, 'isatty', lambda: False)())} "
        f"columns={shutil.get_terminal_size((0, 0)).columns}",
        "env: " + " ".join(f"{k}={os.environ.get(k)!r}" for k in CRASH_ENV_KEYS),
        f"report: {report_path}",
        f"logs in {logs}: {_log_listing(logs)}",
        "",
        "".join(traceback.format_exception(exc)),
    ])
    for folder in (logs, Path(tempfile.gettempdir())):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (f"crash-{stamp}.log" if folder == logs else f"lerobot-doctor-crash-{stamp}.log")
            path.write_text(text, encoding="utf-8")
            return path
        except OSError:
            continue
    return Path("(could not write a crash log anywhere)")


def _pause_if_window_would_close():
    """Only when nothing else keeps the window open: a Windows console with no launcher around it.
    doctor.bat / doctor.ps1 set DOCTOR_LAUNCHER and pause themselves; a Linux/macOS terminal stays."""
    if platform.system() != "Windows" or os.environ.get("DOCTOR_LAUNCHER"):
        return
    try:
        if sys.stdin and sys.stdin.isatty():
            input(bi("按回车关闭窗口", "Press Enter to close") + " > ")
    except (EOFError, OSError):
        pass


def report_crash(con: Console, exc: BaseException, role: str, report_path: Path | None) -> int:
    """The last line of defence. Nothing in here may raise."""
    try:
        con.stop_heartbeat()
    except Exception:  # noqa: BLE001
        pass
    try:
        path = write_crash_log(exc, role, report_path)
    except Exception:  # noqa: BLE001 - the log writer must never be the second failure
        path = Path("(could not write a crash log)")
    try:
        traceback.print_exception(exc, file=sys.stderr)   # the full traceback stays on screen
        sys.stderr.flush()
        con.line("")
        con.item("bad", "体检程序自己出错了（这是工具的问题，不是你电脑的结论）",
                 "The tool itself crashed (a tool problem, not a verdict about your machine)")
        con.line(f"     {bi('错误', 'error')}: {type(exc).__name__}: {str(exc)[:300]}")
        con.line(f"     {bi('位置', 'where')}: {crash_location(exc)}")
        con.line(f"     {bi('崩溃日志', 'crash log')}: {path}")
        if report_path and report_path.exists():
            con.line(f"     {bi('报告', 'report')}: {report_path}")
        con.line(f"     {bi('请把上面的文件贴到这里', 'please attach the file(s) above here')}: {ISSUES_URL}")
    except Exception:  # noqa: BLE001 - even the pretty printer failed: plain ASCII, no formatting
        print(f"\nlerobot-doctor crashed: {type(exc).__name__}: {str(exc)[:300]}\ncrash log: {path}\nreport it: {ISSUES_URL}")
    _pause_if_window_would_close()
    return EXIT_CRASH


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version",
                   version=f"LeRobot Doctor {TOOL_VERSION} · lerobot {LEROBOT_VERSION} · python {sys.version.split()[0]} ({sys.executable})")
    p.add_argument("--specs-only", action="store_true", help="stage 1 only; no install, no download")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask before downloading")
    p.add_argument("--skip-install", action="store_true", help="reuse ~/lerobot-doctor/.venv as is")
    p.add_argument("--install-only", action="store_true", help="stop after the environment is built and importable")
    p.add_argument("--use-current-env", action="store_true", help="run the probes with the Python that runs this file")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None, help="override the test device")
    p.add_argument("--vram-cap", type=float, default=None, help="pretend the GPU has only this many GB (testing)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="3D page port (127.0.0.1 only)")
    p.add_argument("--levels", default=None, help="comma-separated level ids to run, e.g. L1,L3 (others are reported as not tested)")
    p.add_argument("--no-sim", action="store_true", help="no 3D page")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    p.add_argument("--ascii", action="store_true", help="ASCII markers instead of emoji")
    p.add_argument("--uninstall", action="store_true", help=f"remove {WORK_DIR}")
    p.add_argument("--worker", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    p.add_argument("--orchestrate", default=None, help=argparse.SUPPRESS)  # path of the report to continue
    return p


def main(argv=None) -> int:
    _make_streams_non_fatal()
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        return worker_main(args.worker)   # a worker's traceback goes to its log; the parent classifies it
    con = Console(ascii_only=True if args.ascii else None)
    try:
        if args.uninstall:   # inside the crash handler: a file still held by an orphaned worker must not end as a bare traceback
            if WORK_DIR.exists():
                shutil.rmtree(WORK_DIR)
                con.line(bi(f"已删除 {WORK_DIR}（环境、日志、报告；Hugging Face 缓存保留）",
                            f"removed {WORK_DIR} (env, logs, reports; Hugging Face cache kept)"))
            else:
                con.line(bi("没有要删的东西", "nothing to remove"))
            return 0
        return _main(args, con)
    except KeyboardInterrupt:
        con.stop_heartbeat()
        con.line("")
        con.line(bi("已中断。已完成的部分在报告 JSON 里。", "Interrupted. Finished parts are in the report JSON."))
        if CURRENT_RUN["report"]:
            con.line(f"  {CURRENT_RUN['report']}")
        return 130
    except Exception as e:  # noqa: BLE001 - anything else is a bug in this tool: log it, say so, keep the window
        report_path = Path(args.orchestrate) if args.orchestrate else CURRENT_RUN["report"]
        return report_crash(con, e, "orchestrator" if args.orchestrate else "launcher", report_path)


def _main(args, con: Console) -> int:
    started = time.monotonic()
    prov = provenance()
    via = " · ".join(x for x in (prov["doctor_tag"] or "", f"via {prov['launcher']}" if prov["launcher"] else "") if x)
    con.line(f"LeRobot Doctor {TOOL_VERSION}{f' ({via})' if via else ''}  ·  "
             f"{bi('目标：LeRobot ' + LEROBOT_VERSION + ' + SO-101', 'target: LeRobot ' + LEROBOT_VERSION + ' + SO-101')}")
    con.start_heartbeat()

    if args.orchestrate:
        # The orchestrator imports lerobot only to show camera frames on the 3D page. Library warnings
        # (torchcodec fallback tracebacks and the like) belong in the logs, not on a beginner's screen.
        import logging
        import warnings
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        CURRENT_RUN["report"] = Path(args.orchestrate)
        report = Report(Path(args.orchestrate), json.loads(Path(args.orchestrate).read_text(encoding="utf-8")))
        py = Path(sys.executable)
        sim = None
        try:
            sim = orchestrate(args, con, py, report)
        finally:
            report.data["seconds"] = round(time.monotonic() - started + report.data.get("seconds", 0))
            report.data["finished_at"] = now_iso()
            report.save()
            con.stop_heartbeat()
        verdicts = evaluate(report.data)
        report.data["verdicts"] = verdicts
        report.save()
        render_report(con, report.data, verdicts, report.path)
        if con.tty and sim is not None:   # only when the page really exists
            try:   # keep the 3D page alive until the user has looked at everything
                input(bi("回车退出（浏览器里的 3D 页面随之关闭）", "Enter to exit (the 3D page closes with it)") + " > ")
            except EOFError:
                pass
        if sim is not None:
            try:
                sim.close()
            except Exception:  # noqa: BLE001 - shutting down; nothing left to protect
                pass
        return 0

    # ---- stage 1 ------------------------------------------------------------------------------
    total_steps = 4 + 2 * len(LEVELS)
    con.step(1, total_steps, "读取电脑规格", "reading machine specs", "几秒 / seconds")
    specs = collect_specs()
    if args.device:
        specs["accelerator"] = args.device
        specs["device_mem_gb"] = specs["ram_gb"] if args.device == "cpu" else specs["device_mem_gb"]
        specs["bf16"] = specs["bf16"] and args.device == "cuda"
    if args.vram_cap and specs["accelerator"] == "cuda":
        specs["device_mem_gb"] = args.vram_cap
    print_specs(con, specs)
    floors = hard_floors(specs)
    est = estimate_table(specs)
    print_estimate(con, est)
    report_path = new_report_path()
    CURRENT_RUN["report"] = report_path
    report = Report(report_path, {"schema": 1, "tool_version": TOOL_VERSION, "lerobot_version": LEROBOT_VERSION, **prov,
                                  "started_at": now_iso(), "specs": specs, "floors": floors, "estimate": est,
                                  "install": {"status": "NOT_RUN"}, "levels": {}})
    if floors:
        con.line("")
        for f in floors:
            con.item("bad", f["zh"], f["en"])
        con.line(bi("硬门槛未过，不安装、不下载。处理后重跑。", "A hard floor failed; nothing is installed or downloaded. Fix it and rerun."))
        con.stop_heartbeat()
        return 1
    if args.specs_only:
        con.stop_heartbeat()
        con.line("")
        con.line(bi(f"只做了第一段。完整体检请不带 --specs-only 重跑。报告：{report_path}", f"Stage 1 only. Rerun without --specs-only for the full check. Report: {report_path}"))
        return 0

    # ---- confirm ------------------------------------------------------------------------------
    con.step(2, total_steps, "确认", "confirm")
    con.line(bi("接下来会：建一个独立的 Python 环境（约 7 GB）→ 装 LeRobot 0.6.1 → 下载样例数据与三个预训练模型（约 13 GB）→ 五个策略逐个真跑。",
                "Next: create a private Python env (~7 GB) -> install LeRobot 0.6.1 -> download sample data and three pretrained models (~13 GB) -> run five policies for real."))
    if specs["accelerator"] == "cpu":
        con.line(bi("全程约 1–2.5 小时：CPU 上每级推理与训练都慢，多数训练探针会到预算即停。中途 Ctrl-C 可停，已完成部分会保留。",
                    "About 1 to 2.5 hours: every inference and training probe is slow on a CPU, and most training probes stop at their budget. Ctrl-C stops; finished parts are kept."))
    else:
        con.line(bi("全程 30–60 分钟，取决于网速与机器。中途 Ctrl-C 可停，已完成部分会保留。",
                    "30-60 minutes depending on network and machine. Ctrl-C stops; finished parts are kept."))
    if not args.yes and con.tty:
        try:
            input(bi("回车继续，Ctrl-C 退出", "Enter to continue, Ctrl-C to quit") + " > ")
        except EOFError:
            pass

    # ---- install ------------------------------------------------------------------------------
    con.step(3, total_steps, f"安装 LeRobot {LEROBOT_VERSION}", f"installing LeRobot {LEROBOT_VERSION}", "5–20 min")
    if args.use_current_env or (args.skip_install and venv_python(WORK_DIR / ".venv").exists()):
        py = Path(sys.executable) if args.use_current_env else venv_python(WORK_DIR / ".venv")
        rc, info, err = import_probe(py)
        if rc != 0 or info is None:
            report.data["install"] = {"status": "FAIL", "reused": True, "reason": "existing env cannot import lerobot/torch/viser",
                                      "stderr": err, "log": ""}
        else:
            report.data["install"] = {"status": "PASS", "reused": True, **info}
            if info.get("lerobot") != LEROBOT_VERSION:   # a stale venv: the report must say what it measured
                report.data["install"]["version_mismatch"] = info.get("lerobot")
                con.item("warn", f"已有环境里的 lerobot 是 {info.get('lerobot')}，不是 {LEROBOT_VERSION}；这次就按它测，报告会注明",
                         f"the existing env has lerobot {info.get('lerobot')}, not {LEROBOT_VERSION}; testing with it, the report says so")
    else:
        py = build_env(con, specs, report.data, args)
    report.save()
    inst = report.data["install"]
    if inst.get("status") != "PASS" or py is None:
        con.item("bad", *install_failure_text(inst))
        report.data["seconds"] = round(time.monotonic() - started)
        report.save()
        con.stop_heartbeat()
        render_report(con, report.data, evaluate(report.data), report.path)
        return 1
    con.item("ok", f"lerobot {inst['lerobot']} · torch {inst['torch']} · cuda {inst['cuda']} · mps {inst['mps']} · torchcodec {inst['torchcodec'] or 'no (pyav)'}",
             "installed and importable")
    if inst.get(specs["accelerator"]) is False and specs["accelerator"] != "cpu":
        con.item("warn", f"torch 看不到 {specs['accelerator']}，改按 CPU 测", f"torch does not see {specs['accelerator']}; testing on CPU")
        specs["accelerator"] = "cpu"
        specs["device_mem_gb"] = specs["ram_gb"]
        specs["bf16"] = False
    report.data["seconds"] = round(time.monotonic() - started)
    report.save()
    con.stop_heartbeat()
    if args.install_only:
        con.line(bi(f"环境已就绪：{py}", f"environment ready: {py}"))
        return 0

    # ---- hand over to the venv ----------------------------------------------------------------
    handover = [str(py), str(Path(__file__).resolve()), "--orchestrate", str(report_path)]
    for flag in ("--no-sim", "--no-browser", "--ascii", "--yes"):
        if getattr(args, flag[2:].replace("-", "_")):
            handover.append(flag)
    if args.device:
        handover += ["--device", args.device]
    if args.vram_cap:
        handover += ["--vram-cap", str(args.vram_cap)]
    handover += ["--port", str(args.port)]
    if args.levels:
        handover += ["--levels", args.levels]
    return subprocess.call(handover, env=child_env())


if __name__ == "__main__":
    raise SystemExit(main())
