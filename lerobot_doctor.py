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
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

# ----------------------------------------------------------------------------------------------
# Constants. Every threshold is named and carries its source; nothing is tuned per machine.
# ----------------------------------------------------------------------------------------------

TOOL_VERSION = "0.1.0"
LEROBOT_VERSION = "0.6.1"
PYTHON_VERSION = "3.12"                         # lerobot 0.6.1: Requires-Python >=3.12
VISER_SPEC = "viser[urdf]==1.1.0"               # same pin as the Season-1 course repo
LEROBOT_EXTRAS = "smolvla,xvla,wallx,diffusion,dataset,feetech,accelerate-dep"
WORK_DIR = Path.home() / "lerobot-doctor"
DEFAULT_PORT = 4604                             # yanshirobotics 46xx range; 127.0.0.1 only

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
DEMO_FOLLOW_ALPHA = 0.35   # first-order low-pass: how fast the simulated joints follow a target
ACT_DEMO_STEPS = 300       # extra ACT training steps so the second demo shows a learning model
ACT_DEMO_BUDGET_S = 300    # ...but never more than five minutes of it (a CPU may need seconds per step)

TRAIN_WARMUP, TRAIN_TIMED = 3, 10
REF_FRAMES = 45_000        # HF hardware guide: 50 episodes x 30 s x 30 fps
REF_EPOCHS = 5             # HF hardware guide: imitation learning converges in 5-10 epochs
LOCAL_OK_H = 2.0           # projected hours: comfortable locally
OVERNIGHT_H = 12.0         # projected hours: one night; above this we say "cloud"

INFER_BUDGET_S = 12 * 60
TRAIN_BUDGET_S = 15 * 60
TRAIN_L1_BUDGET_S = 25 * 60
HEARTBEAT_SILENCE_S = 2.0
NON_TTY_HEARTBEAT_S = 10.0
CHILD_POLL_S = 0.05
UV_HTTP_TIMEOUT_S = 600    # uv's default 30 s drops 600 MB CUDA wheels on slow links
INSTALL_ATTEMPTS = 3       # network hiccups are the most common student failure; uv caches finished wheels
HF_HTTP_TIMEOUT_S = 60
HF_HTTP_ATTEMPTS = 2

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
    def __init__(self, lid, policy, group, source, weights, extra, batches, large=False, label=None):
        self.id = lid
        self.policy = policy
        self.group = group
        self.source = source          # lerobot-format checkpoint loaded with from_pretrained, or None (built from config)
        self.weights = weights        # HF repo whose size/params drive downloads and the weight floor, or None (from scratch)
        self.extra = extra
        self.batches = batches
        self.large = large            # bfloat16 + gradient checkpointing where the policy config supports it
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
STATUS_FLOOR = {"SKIPPED_FLOOR"}
STATUS_NOT_RUN = {"TIMEOUT", "FAIL_DEP", "FAIL_DOWNLOAD", "BLOCKED_GATED", "FAIL_CRASH", "NOT_RUN"}
MEMORY_FAILS = {"FAIL_OOM", "FAIL_RAM", "SKIPPED_FLOOR"}


def bi(zh: str, en: str) -> str:
    """One bilingual line. Chinese first, English after a separator."""
    return f"{zh}  |  {en}"


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
        self._lock = threading.RLock()
        self._last_output = time.monotonic()
        self._activity = ""
        self._activity_since = time.monotonic()
        self._transient_len = 0
        self._stop = threading.Event()
        self._thread = None
        self._last_nontty_beat = 0.0

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

    def transient(self, text: str):
        """A status line that gets overwritten (tty) or throttled to one line per 10 s (non-tty)."""
        with self._lock:
            now = time.monotonic()
            if self.tty:
                self._clear_transient()
                self.stream.write("\r" + text)
                self.stream.flush()
                self._transient_len = len(text) + 2
            elif now - self._last_nontty_beat >= NON_TTY_HEARTBEAT_S:
                self.stream.write(text + "\n")
                self.stream.flush()
                self._last_nontty_beat = now
            self._last_output = now

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
                text = f"{spin[i]} {self._activity} … {fmt_duration(elapsed)}"
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
        specs["os"] = f"Windows {rel} ({ver})"
        specs["windows_release"] = rel
        try:
            import winreg  # type: ignore
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            specs["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:  # noqa: BLE001 - best effort on a foreign registry
            specs["cpu"] = platform.processor()
        specs["ram_gb"] = _windows_ram_gb()
        ps = run_cmd(["powershell", "-NoProfile", "-Command",
                      "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        specs["other_gpus"] = [g.strip() for g in ps.splitlines() if g.strip() and "NVIDIA" not in g]
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


def weight_floor(level: Level, params: int | None, dtype: str, device_mem_gb: float | None) -> dict | None:
    """Pure. The only per-level floor: the weights alone do not fit the device memory."""
    if not params or not device_mem_gb:
        return None
    need_gb = params * BYTES_PER_PARAM[dtype] / 1e9
    if need_gb > device_mem_gb:
        return {"need_gb": round(need_gb, 1), "have_gb": device_mem_gb, "dtype": dtype,
                "zh": f"权重 {need_gb:.1f} GB（{params/1e9:.2f}B 参数 × {dtype}）> 设备内存 {device_mem_gb} GB",
                "en": f"weights {need_gb:.1f} GB ({params/1e9:.2f}B params x {dtype}) > device memory {device_mem_gb} GB"}
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
    tok = hf_token()
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

def print_specs(con: Console, specs: dict):
    con.line(bi("系统体检 · 第一段（只读，不安装任何东西）", "System check, stage 1 (read-only, installs nothing)"))
    rows = [
        ("系统 OS", specs["os"] + (" · WSL" if specs.get("wsl") else "")),
        ("架构 Arch", specs["arch"]),
        ("处理器 CPU", f"{specs['cpu']} · {specs['cpu_count']} threads"),
        ("内存 RAM", f"{specs['ram_gb']} GB" if specs["ram_gb"] else "?"),
        ("磁盘剩余 Free disk", f"{specs['disk_free_gb']} GB"),
        ("Python", specs["python"]),
        ("ffmpeg", "yes" if specs["ffmpeg"] else "no (video decode falls back to pyav)"),
    ]
    for g in specs["nvidia"]:
        rows.append(("显卡 GPU (NVIDIA)", f"{g['name']} · {g['vram_gb']} GB · driver {g['driver']}"
                                          + (f" · compute {g['compute_cap']}" if g['compute_cap'] else "")))
    for g in specs["other_gpus"]:
        rows.append(("显卡 GPU (other)", g))
    acc = specs["accelerator"]
    acc_text = {"cuda": f"CUDA ({specs['device_mem_gb']} GB VRAM, bf16 {'yes' if specs['bf16'] else 'no'})",
                "mps": f"Apple MPS ({specs['device_mem_gb']} GB unified memory)",
                "cpu": "CPU only"}[acc]
    rows.append(("测试用设备 Device for tests", acc_text))
    width = max(dwidth(r[0]) for r in rows)
    for k, v in rows:
        con.line(f"  {pad(k, width)}  {v}")
    notes = {
        "driver_too_old": ("NVIDIA 驱动低于 570.86：这次按 CPU 测，GPU 一列全部记「未测」。升级驱动后重跑。",
                           "NVIDIA driver below 570.86: testing on CPU; the GPU column is 'not tested'. Update the driver and rerun."),
        "rosetta": ("Python 跑在 Rosetta 下（x86 版），MPS 不可用。装 arm64 版 Python 后重跑。",
                    "Python runs under Rosetta (x86 build); MPS unavailable. Install arm64 Python and rerun."),
        "macos_too_old": ("macOS 低于 12.3，MPS 不可用，按 CPU 测。", "macOS below 12.3: no MPS, testing on CPU."),
        "intel_mac": ("Intel Mac：无加速器，按 CPU 测；torchcodec 无轮子，LeRobot 会自动改用 pyav。",
                      "Intel Mac: no accelerator, testing on CPU; no torchcodec wheel, LeRobot falls back to pyav."),
        "non_nvidia_gpu": ("检测到非 NVIDIA 独显：本工具不覆盖 AMD/Intel GPU 路线，按 CPU 测。",
                           "Non-NVIDIA GPU found: this tool does not cover AMD/Intel GPU paths; testing on CPU."),
    }
    if specs.get("accelerator_note") in notes:
        con.item("warn", *notes[specs["accelerator_note"]])
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
    for cand in (Path.home() / ".local" / "bin" / "uv", Path.home() / ".cargo" / "bin" / "uv",
                 Path.home() / ".local" / "bin" / "uv.exe"):
        if cand.exists():
            return str(cand)
    return None


def stream_process(cmd, con: Console, log_path: Path, env=None, cwd=None, timeout=None, on_line=None) -> int:
    """Run a command, mirror its output to the console (and a log), keep the heartbeat alive."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(map(str, cmd))}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                                bufsize=0)
        start = time.monotonic()
        buf = b""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1)
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(CHILD_POLL_S)
                continue
            buf += chunk
            while True:
                m = re.search(rb"[\r\n]", buf)
                if not m:
                    break
                seg, sep, buf = buf[:m.start()], buf[m.start():m.end()], buf[m.end():]
                text = seg.decode("utf-8", errors="replace")
                log.write(text + "\n")
                if on_line and on_line(text, sep == b"\r"):
                    continue
                if sep == b"\r":
                    con.transient(text[-200:])
                elif text.strip():
                    con.raw(text + "\n")
            if timeout and time.monotonic() - start > timeout:
                kill_tree(proc)
                log.write("\n[lerobot-doctor] TIMEOUT\n")
                return -999
        if buf.strip():
            text = buf.decode("utf-8", errors="replace")
            log.write(text + "\n")
            if not (on_line and on_line(text, False)):
                con.raw(text + "\n")
        return proc.wait()


def kill_tree(proc: subprocess.Popen):
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except OSError:
        pass


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
    if not py.exists():
        con.activity(bi("创建虚拟环境", "creating virtualenv"))
        rc = stream_process([uv, "venv", str(venv), "--python", PYTHON_VERSION, "--seed"], con, logs / "install.log")
        if rc != 0:
            install.update(status="FAIL_CRASH", reason=f"uv venv exited {rc}", log=str(logs / "install.log"))
            return None
    spec = f"lerobot[{LEROBOT_EXTRAS}]=={LEROBOT_VERSION}"
    cmd = [uv, "pip", "install", "--python", str(py), spec, VISER_SPEC]
    if backend != "default":
        cmd += ["--torch-backend", backend]
    con.activity(bi("安装 LeRobot（uv 会打印自己的进度）", "installing LeRobot (uv prints its own progress)"))
    t0 = time.monotonic()
    env = dict(os.environ, UV_HTTP_TIMEOUT=str(UV_HTTP_TIMEOUT_S))
    for attempt in range(1, INSTALL_ATTEMPTS + 1):
        rc = stream_process(cmd, con, logs / "install.log", env=env)
        if rc == 0:
            break
        tail = (logs / "install.log").read_text(encoding="utf-8", errors="replace")[-4000:].lower()
        if attempt < INSTALL_ATTEMPTS and ("timeout" in tail or "timed out" in tail or "connection" in tail):
            con.item("warn", f"下载超时，重试 {attempt}/{INSTALL_ATTEMPTS - 1}（已下好的包不重下）",
                     f"download timed out, retry {attempt}/{INSTALL_ATTEMPTS - 1} (finished packages are cached)")
            continue
        break
    install["seconds"] = round(time.monotonic() - t0)
    install["torch_backend"] = backend
    install["log"] = str(logs / "install.log")
    if rc != 0:
        install.update(status="FAIL", reason=f"uv pip install exited {rc}")
        return None
    probe = subprocess.run([str(py), "-c", IMPORT_PROBE], capture_output=True, text=True, timeout=600)
    if probe.returncode != 0:
        install.update(status="FAIL", reason="import failed", stderr=probe.stderr[-2000:])
        (logs / "install.log").open("a", encoding="utf-8").write(probe.stderr)
        return None
    install.update(status="PASS", **json.loads(probe.stdout.strip().splitlines()[-1]))
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
    # floor 1: weights alone do not fit
    w = weights.get(level.id) or {}
    fl = weight_floor(level, w.get("params"), dtype, specs.get("device_mem_gb"))
    if fl:
        return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": "weights_exceed_memory", **fl}
    # floor 2: memory monotonic - a smaller level already failed on memory (same device, same dtype)
    my_params = w.get("params") or 0
    for other in LEVELS:
        if other.id == level.id:
            continue
        r = results.get(other.id, {}).get("infer") or {}
        if r.get("status") in ("FAIL_OOM", "FAIL_RAM") and (weights.get(other.id) or {}).get("params", 0) <= my_params \
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
    for other in LEVELS:   # memory monotonic for training too
        if other.id == level.id:
            continue
        r = results.get(other.id, {}).get("train") or {}
        if r.get("status") in ("FAIL_OOM", "FAIL_RAM") and LEVELS.index(other) < LEVELS.index(level) \
                and r.get("params", 0) <= (inf.get("params") or 0):
            return {"status": "SKIPPED_FLOOR", "evidence": "floor", "reason": f"smaller_level_oom:{other.id}",
                    "zh": f"{other.id} {other.label} batch 1 都训不了，本级更大", "en": f"{other.id} {other.label} failed even at batch 1; this level is larger"}
    return None


def classify_exit(returncode: int, log_text: str) -> str:
    """Map a dead child to a status from its exit code and log."""
    low = log_text.lower()
    if "out of memory" in low or "outofmemoryerror" in low or "mps backend out of memory" in low:
        return "FAIL_OOM"
    if returncode in (-9, 137) or returncode == 3221225477 or returncode == -1073741819:  # SIGKILL / 0xC0000005
        return "FAIL_RAM"
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


# ----------------------------------------------------------------------------------------------
# Verdicts (pure; unit-tested).
# ----------------------------------------------------------------------------------------------

def infer_verdict(level: Level, r: dict) -> dict:
    st = r.get("status", "NOT_RUN")
    ms = r.get("latency_ms")
    bud = r.get("budget_ms")
    demo = r.get("demo") or {}
    demo_txt = ""
    if demo.get("status") == "PASS":
        demo_txt = (f"；模拟执行 {DEMO_SECONDS} s ✅，控制 {demo.get('hz', 0):.0f} Hz",
                    f"; simulated task {DEMO_SECONDS} s OK at {demo.get('hz', 0):.0f} Hz")
    if st == "PASS":
        return {"mark": "ok", "evidence": "measured",
                "zh": f"本地实时推理（每块 {ms:.0f} ms，预算 {bud:.0f} ms）" + (demo_txt[0] if demo_txt else ""),
                "en": f"real-time local inference ({ms:.0f} ms per chunk, budget {bud:.0f} ms)" + (demo_txt[1] if demo_txt else "")}
    if st == "MARGINAL":
        return {"mark": "warn", "evidence": "measured",
                "zh": f"本地可推理但接近上限（每块 {ms:.0f} ms，预算 {bud:.0f} ms）；建议相机降到 320×240 或加长 chunk" + (demo_txt[0] if demo_txt else ""),
                "en": f"local inference works but near the limit ({ms:.0f} ms per chunk, budget {bud:.0f} ms); lower cameras to 320x240 or lengthen the chunk" + (demo_txt[1] if demo_txt else "")}
    if st == "TOO_SLOW":
        return {"mark": "bad", "evidence": "measured",
                "zh": f"本地推理达不到实时（每块 {ms:.0f} ms，预算 {bud:.0f} ms）",
                "en": f"local inference is not real-time ({ms:.0f} ms per chunk, budget {bud:.0f} ms)"}
    if st in ("FAIL_OOM", "FAIL_RAM"):
        what = "显存" if r.get("device") == "cuda" else "内存"
        return {"mark": "bad", "evidence": "measured",
                "zh": f"本地装不下（加载时{what}耗尽）", "en": f"does not fit locally (ran out of {'VRAM' if r.get('device') == 'cuda' else 'memory'} while loading)"}
    if st == "SKIPPED_FLOOR":
        return {"mark": "bad", "evidence": "floor", "zh": f"本地装不下：{r.get('zh', '')}", "en": f"does not fit locally: {r.get('en', '')}"}
    reasons = {
        "BLOCKED_GATED": ("未测：这个模型的仓库要先在 Hugging Face 上同意许可并登录", "not tested: this model's repo needs a Hugging Face login and license acceptance"),
        "FAIL_DEP": ("未测：依赖没装上（见日志）", "not tested: a dependency is missing (see log)"),
        "FAIL_DOWNLOAD": ("未测：下载失败（网络）", "not tested: download failed (network)"),
        "TIMEOUT": ("未测：预算时间内没跑完（磁盘或网络极慢），见日志", "not tested: did not finish within budget (very slow disk or network), see log"),
        "FAIL_CRASH": ("未测：程序异常，见日志。这是工具或环境的问题，不是你电脑的结论", "not tested: the probe crashed, see log. That is a tool/environment problem, not a verdict about your machine"),
        "NOT_RUN": (r.get("zh", "未测"), r.get("en", "not tested")),
    }
    zh, en = reasons.get(st, reasons["NOT_RUN"])
    return {"mark": "skip", "evidence": "not_run", "zh": zh, "en": en}


def train_verdict(level: Level, r: dict) -> dict:
    st = r.get("status", "NOT_RUN")
    if st == "PASS":
        h = r["hours"]
        b = r["batch"]
        s = r["update_s"]
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
        v["evidence"] = "measured"
        return v
    if st in ("FAIL_OOM", "FAIL_RAM"):
        return {"mark": "bad", "evidence": "measured", "cloud": True,
                "zh": "本地训不了（batch 1 也装不下）→ 需要上云", "en": "cannot train locally (even batch 1 does not fit) -> needs cloud"}
    if st == "SKIPPED_FLOOR":
        return {"mark": "bad", "evidence": "floor", "cloud": True,
                "zh": f"本地训不了：{r.get('zh', '')}", "en": f"cannot train locally: {r.get('en', '')}"}
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
    r2 = [lv for lv in ordered if infer_ok(levels_v[lv.id]) and train_cloud(levels_v[lv.id])]
    if r2:
        k = r2[-1]
        return {"rule": "R2", "level": k.id,
                "zh": f"录数据在本地 → 上云训练 {k.label} → 权重拿回本地推理。",
                "en": f"Record locally -> train {k.label} in the cloud -> bring the weights back and run locally."}
    any_infer_ok = [lv for lv in ordered if infer_ok(levels_v[lv.id])]
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
    if specs["system"] == "Darwin":
        notes.append(("键盘遥操作要给终端「辅助功能」权限；课程的相机脚本是 Linux 专用，mac 上用 lerobot-find-cameras。",
                      "Keyboard teleop needs Accessibility permission for the terminal; the course camera script is Linux-only, use lerobot-find-cameras on mac."))
    if specs["system"] == "Windows":
        notes.append(("串口叫 COMx，不是 /dev/ttyACM0；课程命令里替换即可。", "Serial ports are COMx, not /dev/ttyACM0; substitute in the course commands."))
    if specs.get("wsl"):
        notes.append(("USB 串口要用 usbipd 转发进 WSL。", "USB serial must be forwarded into WSL with usbipd."))
    if install.get("torchcodec") is None and install.get("status") == "PASS":
        notes.append(("torchcodec 不可用，视频解码走 pyav：正常，只是慢一点。", "torchcodec unavailable, video decoding uses pyav: fine, just slower."))
    return notes


def evaluate(report: dict) -> dict:
    """Turn raw results into verdicts. Pure."""
    install = report.get("install", {})
    out = {"install": install.get("status"), "levels": {}, "basics": None, "route": None, "notes": []}
    if install.get("status") != "PASS":
        out["route"] = {"rule": "R0", "zh": "lerobot 0.6.1 装不上，见 install.log。", "en": "lerobot 0.6.1 did not install, see install.log."}
        return out
    ds = report.get("dataset", {})
    if ds.get("status") != "PASS":
        out["route"] = {"rule": "R0", "zh": "样例数据下载失败（网络问题），硬件没有得出任何结论。", "en": "Sample dataset download failed (network); no hardware verdict."}
        out["basics"] = {"mark": "ok"}
        return out
    out["basics"] = {"mark": "ok"}
    out["notes"] = basics_notes(report["specs"], install)
    for lv in LEVELS:
        res = report.get("levels", {}).get(lv.id, {})
        out["levels"][lv.id] = {"infer": infer_verdict(lv, res.get("infer") or {"status": "NOT_RUN"}),
                                "train": train_verdict(lv, res.get("train") or {"status": "NOT_RUN"})}
    out["route"] = route_verdict(out["levels"])
    return out


def render_report(con: Console, report: dict, verdicts: dict):
    con.line("")
    con.line("=" * 78)
    con.line(bi("体检报告", "Report"))
    con.line("=" * 78)
    specs = report["specs"]
    dev = {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}[specs["accelerator"]]
    con.line(f"  {specs['os']} · {dev} {specs.get('device_mem_gb')} GB · RAM {specs['ram_gb']} GB · "
             f"{bi('总耗时', 'total')} {fmt_duration(report.get('seconds', 0))}")
    inst = report.get("install", {})
    con.item("ok" if inst.get("status") == "PASS" else "bad",
             f"LeRobot {LEROBOT_VERSION} 安装并可导入" if inst.get("status") == "PASS" else "LeRobot 0.6.1 装不上",
             f"LeRobot {LEROBOT_VERSION} installed and importable" if inst.get("status") == "PASS" else "LeRobot 0.6.1 failed to install")
    if verdicts.get("basics"):
        con.item("ok", "组装 / 标定 / 遥操作 / 录数据：可以", "assemble / calibrate / teleoperate / record: yes")
        for zh, en in verdicts["notes"]:
            con.line(f"      · {bi(zh, en)}")
    if verdicts["levels"]:
        con.line("")
        con.line(bi("各级实测（⛔ 只来自实测或硬门槛；— 是未测）", "Per level (a cross comes only from a measurement or a hard floor; a dash means not tested)"))
        for lv in LEVELS:
            v = verdicts["levels"][lv.id]
            con.line(f"  {lv.id} {lv.label} ({lv.group})")
            con.line(f"      {bi('推理', 'inference')} {con.mark(v['infer']['mark'])} {v['infer']['zh']}")
            con.line(f"      {' ' * len('推理 | inference')} {v['infer']['en']}")
            con.line(f"      {bi('训练', 'training ')} {con.mark(v['train']['mark'])} {v['train']['zh']}")
            con.line(f"      {' ' * len('训练 | training ')} {v['train']['en']}")
            est = (report.get("estimate") or {}).get(lv.id, {}).get("train_estimate")
            measured = v["train"].get("cloud")
            if est and measured is not None and (est == "cloud") != measured:
                words = {"local": "本地可训 / train locally", "tight": "勉强 / tight", "cloud": "需上云 / cloud"}
                con.line(f"      * {bi('预估表曾说', 'the estimate said')} 「{words[est]}」，{bi('以实测为准', 'the measurement wins')}")
    con.line("")
    r = verdicts["route"]
    con.line(bi("SO-101 路线", "SO-101 route") + f" [{r['rule']}]")
    con.line(f"  {r['zh']}")
    con.line(f"  {r['en']}")


# ----------------------------------------------------------------------------------------------
# Orchestrator (runs inside the venv): dataset, URDF, sim page, ladder, report.
# ----------------------------------------------------------------------------------------------

class Report:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def child_env() -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


_PROGRESS_RE = re.compile(r"\d+%\||\d+(\.\d+)?\s?[kMG]?B/s|it/s|Downloading|Fetching")


def is_download_progress(text: str) -> bool:
    """tqdm / huggingface_hub progress lines: the only child stdout a beginner should see."""
    return bool(_PROGRESS_RE.search(text))


def run_worker(py: Path, con: Console, log_path: Path, worker_args: list[str], budget_s: int,
               progress_label: str, on_joints=None) -> dict:
    """Run `lerobot_doctor.py --worker ...` in the venv and collect its @@result."""
    result_holder = {}
    state = {"progress": ""}

    def on_line(text: str, is_cr: bool) -> bool:
        if text.startswith("@@result "):
            try:
                result_holder.update(json.loads(text[len("@@result "):]))
            except json.JSONDecodeError:
                pass
            return True
        if text.startswith("@@progress "):
            _, stage, i, n, *extra = text.split(" ", 4)
            note = extra[0] if extra else ""
            con.transient(f"  {con.mark('run')} {progress_label} {stage} {i}/{n} {note}")
            return True
        if text.startswith("@@activity "):
            con.activity(text[len("@@activity "):])
            return True
        if text.startswith("@@event "):
            _, name, *payload = text.split(" ", 2)
            body = payload[0] if payload else ""
            if name == "oom":
                con.line(f"  {con.mark('warn')} {body}")
            elif name == "note":
                con.line(f"  {con.mark('info')} {body}")
            return True
        if text.startswith("@@joints "):
            if on_joints:
                on_joints(text[len("@@joints "):])
            return True
        # Everything else from the child goes to the log only, except download progress bars:
        # a traceback or a library warning on a beginner's screen reads as "it broke".
        if is_download_progress(text):
            con.transient(text[-200:]) if is_cr else con.raw(text + "\n")
        return True

    cmd = [str(py), str(Path(__file__).resolve()), "--worker", *worker_args]
    t0 = time.monotonic()
    rc = stream_process(cmd, con, log_path, env=child_env(), timeout=budget_s, on_line=on_line)
    seconds = round(time.monotonic() - t0, 1)
    if rc == -999:
        return {"status": "TIMEOUT", "evidence": "not_run", "seconds": seconds, "log": str(log_path)}
    if result_holder and rc == 0:
        result_holder.setdefault("seconds", seconds)
        result_holder["log"] = str(log_path)
        return result_holder
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:] if log_path.exists() else ""
    status = classify_exit(rc, tail)
    if result_holder.get("status"):   # the worker classified its own failure before exiting non-zero
        status = result_holder["status"]
    err = ""
    for line in reversed(tail.splitlines()):
        if "Error" in line or "error" in line:
            err = line.strip()[:300]
            break
    return {"status": status, "evidence": "measured" if status in STATUS_MEASURED_FAIL else "not_run",
            "returncode": rc, "seconds": seconds, "log": str(log_path), "error": err, **{k: v for k, v in result_holder.items() if k != "status"}}


def orchestrate(args, con: Console, py: Path, report: Report) -> None:
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
        con.item("bad", f"样例数据没下下来：{ds.get('error') or ds.get('status')}", f"sample dataset failed: {ds.get('error') or ds.get('status')}")
        return
    con.item("ok", f"{ds['frames']} 帧 · 指令「{ds['task']}」· 解码后端 {ds['video_backend']}",
             f"{ds['frames']} frames, task '{ds['task']}', decoder {ds['video_backend']}")

    # ---- sim page -----------------------------------------------------------------------------
    sim = None
    if not args.no_sim:
        con.activity(bi("准备 3D 模拟页面", "preparing the 3D simulation page"))
        try:
            sim = SimPage(port=args.port, urdf_dir=WORK_DIR / "models" / "so101", dataset_root=WORK_DIR / "dataset",
                          con=con, open_browser=not args.no_browser)
            con.item("ok", f"3D 模拟页面：http://127.0.0.1:{args.port}（浏览器应已自动打开）", f"3D simulation page: http://127.0.0.1:{args.port} (a browser tab should have opened)")
        except Exception as e:  # noqa: BLE001 - the page is a bonus, never blocks the probes
            con.item("warn", f"3D 页面没起来（{type(e).__name__}: {str(e)[:120]}），探针照跑", f"3D page failed ({type(e).__name__}); probes continue")
            sim = None

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
        if sim:
            sim.on_joints(payload)

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
        if sim:
            sim.begin_level(lv.label)
        con.activity(bi(f"{lv.label}：下载 / 加载模型", f"{lv.label}: downloading / loading model"))
        wargs = ["infer", "--level", lv.id, "--device", device, "--dtype", dtype]
        if args.vram_cap:
            wargs += ["--vram-cap", str(args.vram_cap)]
        r = run_worker(py, con, logs / f"{lv.id}-infer.log", wargs, INFER_BUDGET_S, lv.label, on_joints=joints_cb)
        r["dtype"] = dtype
        r["device"] = device
        r["params"] = (weights.get(lv.id) or {}).get("params") or r.get("params")
        results[lv.id]["infer"] = r
        v = infer_verdict(lv, r)
        con.item(v["mark"], v["zh"], v["en"])
        if sim:
            sim.end_level(lv.label, r)
        report.save()

    # ---- training ladder --------------------------------------------------------------------
    for lv in LEVELS:
        next_step(f"{lv.label} 训练", f"{lv.label} training", "2–15 min")
        pre = None if lv.id in selected else dict(filtered)
        pre = pre or train_precheck(lv, results)
        if pre:
            results[lv.id]["train"] = pre
            con.item("skip" if pre["evidence"] == "not_run" else "bad", pre["zh"], pre["en"])
            report.save()
            continue
        con.activity(bi(f"{lv.label}：加载模型准备训练", f"{lv.label}: loading model for training"))
        wargs = ["train", "--level", lv.id, "--device", device, "--dtype", dtype]
        if args.vram_cap:
            wargs += ["--vram-cap", str(args.vram_cap)]
        budget = TRAIN_L1_BUDGET_S if lv.id == "L1" else TRAIN_BUDGET_S
        if lv.id == "L1" and sim:
            sim.begin_level(lv.label + " " + bi("（训练后）", "(after training)"))
        r = run_worker(py, con, logs / f"{lv.id}-train.log", wargs, budget, lv.label, on_joints=joints_cb)
        r["params"] = (weights.get(lv.id) or {}).get("params") or r.get("params")
        results[lv.id]["train"] = r
        if r.get("status") == "PASS":
            r["hours"] = round(projected_hours(r["update_s"], r["batch"]), 1)
        v = train_verdict(lv, r)
        con.item(v["mark"], v["zh"], v["en"])
        if lv.id == "L1" and r.get("demo"):
            results[lv.id]["infer"]["demo_after_training"] = r["demo"]
        report.save()

    if sim:
        sim.finish()


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
        with urllib.request.urlopen(base + remote, timeout=60) as resp:
            dst.write_bytes(resp.read())
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
            threading.Thread(target=lambda: webbrowser.open(f"http://127.0.0.1:{port}"), daemon=True).start()

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

    emit("activity", bi("下载样例数据（HF 会显示进度条）", "downloading sample data (HF shows a progress bar)"))
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
        emit("note", f"simulating a {cap_gb} GB GPU (memory fraction {frac:.2f})")


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
            emit("note", f"camera rename for {level.label}: {rename_map}")
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


def _observation(ds, idx: int, state, device, task: str, robot_type: str):
    """One inference observation: recorded camera frames + the simulated arm's joint state."""
    import torch
    item = ds[idx]
    obs = {}
    for key in ds.meta.camera_keys:
        img = item[key]
        if img.dtype == torch.uint8:
            img = img.to(torch.float32) / 255.0
        obs[key] = img.unsqueeze(0).to(device)
    obs["observation.state"] = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    obs["task"] = task
    obs["robot_type"] = robot_type
    return obs


def worker_infer(level: Level, device_name: str, dtype: str, vram_cap: float | None,
                 checkpoint: Path | None = None, demo_only: bool = False) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = _torch_device(device_name)
    _apply_vram_cap(device, vram_cap)
    root = WORK_DIR / "dataset"
    ds = LeRobotDataset(DATASET_REPO, root=root, episodes=[DEMO_EPISODE])
    task = str(ds.meta.tasks.index[0])
    robot_type = ds.meta.robot_type or "so101_follower"

    emit("activity", bi(f"{level.label}：下载 / 加载模型到 {device_name}", f"{level.label}: downloading / loading the model to {device_name}"))
    t_load = time.monotonic()
    try:
        policy, pre, post, _ = load_level_policy(level, device, dtype, ds.meta, for_training=False, checkpoint=checkpoint)
    except BaseException as e:  # noqa: BLE001 - classified below
        if _is_oom(e):
            emit_result({"status": "FAIL_OOM", "evidence": "measured", "phase": "load", "peak_gb": _peak_mem_gb(device)})
            return {"status": "FAIL_OOM"}
        raise
    load_s = round(time.monotonic() - t_load, 1)
    params = sum(p.numel() for p in policy.parameters())
    n_action_steps = int(getattr(policy.config, "n_action_steps", 1))
    result = {"status": "PASS", "evidence": "measured", "params": params, "load_s": load_s,
              "n_action_steps": n_action_steps, "device": device_name, "dtype": dtype}

    state0 = ds[0]["observation.state"].numpy().tolist()
    if not demo_only:
        # ---- timing: predict_action_chunk is the real forward pass ----------------------
        is_gpu = device.type == "cuda"
        warm, timed = (WARMUP_GPU, TIMED_GPU) if is_gpu else (WARMUP_CPU, TIMED_CPU)
        budget = n_action_steps / CONTROL_FPS
        lat = []
        emit("activity", bi(f"{level.label}：预热", f"{level.label}: warm-up"))
        try:
            with torch.inference_mode():
                # select_action on an empty queue = exactly one forward pass (predict_action_chunk)
                # plus a queue pop. reset() before each call keeps every call a real forward, and this
                # is the same code path lerobot-record uses on the robot.
                for i in range(warm):
                    obs = _observation(ds, i, state0, device, task, robot_type)
                    policy.reset()
                    policy.select_action(pre(obs))
                    _sync(device)
                    emit("progress", "warmup", i + 1, warm)
                for i in range(timed):
                    obs = _observation(ds, i, state0, device, task, robot_type)
                    policy.reset()
                    t0 = time.perf_counter()
                    policy.select_action(pre(obs))
                    _sync(device)
                    dt = time.perf_counter() - t0
                    lat.append(dt)
                    emit("progress", "timing", i + 1, timed, f"median {statistics.median(lat)*1000:.0f} ms")
                    if i == 0 and not is_gpu and dt > HOPELESS_RATIO * budget:
                        emit("note", f"one forward pass already {dt/budget:.0f}x over the {budget:.2f} s budget; stopping the timing here")
                        break
        except BaseException as e:  # noqa: BLE001
            if _is_oom(e):
                emit_result({**result, "status": "FAIL_OOM", "phase": "forward", "peak_gb": _peak_mem_gb(device)})
                return {"status": "FAIL_OOM"}
            raise
        med = statistics.median(lat)
        status, budget = realtime_status(med, n_action_steps)
        result.update(status=status, latency_ms=round(med * 1000, 1), p95_ms=round(sorted(lat)[int(0.95 * (len(lat) - 1))] * 1000, 1),
                      budget_ms=round(budget * 1000), timed_calls=len(lat), peak_gb=_peak_mem_gb(device))
    # ---- simulated task ------------------------------------------------------------------
    if result["status"] in ("PASS", "MARGINAL") or demo_only:
        result["demo"] = run_demo(policy, pre, post, ds, device, task, robot_type, state0)
    emit_result(result)
    return result


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_demo(policy, pre, post, ds, device, task, robot_type, state0) -> dict:
    """Half-closed loop: recorded frames in, model actions drive a kinematic arm whose joint
    state is fed back as observation.state. Joint targets stream to the parent as @@joints."""
    import torch
    n_steps = DEMO_SECONDS * CONTROL_FPS
    n_avail = min(len(ds), n_steps)
    sim = list(state0)
    policy.reset()
    errs = []
    t_start = time.monotonic()
    last_lat = 0.0
    done = 0
    emit("activity", bi("模拟执行样例任务", "running the simulated task"))
    with torch.inference_mode():
        for t in range(n_avail):
            tick = time.monotonic()
            obs = _observation(ds, t, sim, device, task, robot_type)
            t0 = time.perf_counter()
            action = policy.select_action(pre(obs))
            action = post(action)
            _sync(device)
            dt = time.perf_counter() - t0
            if dt > 0.005:
                last_lat = dt   # a real forward happened (queue refill); pop-from-queue is ~0
            target = action[0].detach().to("cpu").float().numpy().tolist()[:6]
            sim = [s + DEMO_FOLLOW_ALPHA * (tg - s) for s, tg in zip(sim, target)]
            human = ds[t]["action"].numpy().tolist()[:6]
            errs.append(sum(abs(a - b) for a, b in zip(target[:5], human[:5])) / 5)
            done = t + 1
            elapsed = time.monotonic() - t_start
            hz = done / elapsed if elapsed > 0 else 0.0
            emit("joints", t, *[f"{x:.3f}" for x in sim], f"{hz:.1f}", f"{last_lat*1000:.1f}")
            if t % 15 == 0:
                emit("progress", "demo", done, n_avail, f"{hz:.0f} Hz")
            if elapsed > DEMO_WALL_CAP_S:
                break
            sleep_for = 1.0 / CONTROL_FPS - (time.monotonic() - tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
    elapsed = time.monotonic() - t_start
    hz = done / elapsed if elapsed else 0.0
    return {"status": "PASS" if done >= n_avail else "TOO_SLOW", "steps": done, "seconds": round(elapsed, 1),
            "hz": round(hz, 1), "mean_abs_err_deg": round(statistics.mean(errs), 2) if errs else None}


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
        emit("activity", bi(f"{level.label}：batch {batch_size} 加载模型", f"{level.label}: loading model for batch {batch_size}"))
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
                policy = None
                _free_cache(device)
                cfg_policy, rename_map = build_level_config(level, device, dtype, meta, for_training=True)
                continue
            raise
    if result["status"] != "PASS":
        result.update(status="FAIL_OOM", evidence="measured")
        emit_result(result)
        return result
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
        emit("note", f"checkpoint not saved: {type(e).__name__}")
    policy.eval()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    demo_ds = LeRobotDataset(DATASET_REPO, root=WORK_DIR / "dataset", episodes=[DEMO_EPISODE])   # single frames, no chunks
    task = str(ds.meta.tasks.index[0])
    state0 = demo_ds[0]["observation.state"].numpy().tolist()
    demo = run_demo(policy, pre, post, demo_ds, device, task, ds.meta.robot_type or "so101_follower", state0)
    demo["trained_steps"] = n_extra + TRAIN_WARMUP + TRAIN_TIMED
    return demo


def worker_main(argv: list[str]) -> int:
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
# CLI
# ----------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = build_parser().parse_args(argv)
    if args.worker is not None:
        return worker_main(args.worker)
    con = Console(ascii_only=True if args.ascii else None)
    if args.uninstall:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
            con.line(bi(f"已删除 {WORK_DIR}（Hugging Face 缓存保留）", f"removed {WORK_DIR} (Hugging Face cache kept)"))
        else:
            con.line(bi("没有要删的东西", "nothing to remove"))
        return 0
    try:
        return _main(args, con)
    except KeyboardInterrupt:
        con.stop_heartbeat()
        con.line("")
        con.line(bi("已中断。已完成的部分在报告 JSON 里。", "Interrupted. Finished parts are in the report JSON."))
        return 130


def _main(args, con: Console) -> int:
    started = time.monotonic()
    con.line(f"LeRobot Doctor {TOOL_VERSION}  ·  {bi('目标：LeRobot ' + LEROBOT_VERSION + ' + SO-101', 'target: LeRobot ' + LEROBOT_VERSION + ' + SO-101')}")
    con.start_heartbeat()

    if args.orchestrate:
        # The orchestrator imports lerobot only to show camera frames on the 3D page. Library warnings
        # (torchcodec fallback tracebacks and the like) belong in the logs, not on a beginner's screen.
        import logging
        import warnings
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        report = Report(Path(args.orchestrate), json.loads(Path(args.orchestrate).read_text(encoding="utf-8")))
        py = Path(sys.executable)
        try:
            orchestrate(args, con, py, report)
        finally:
            report.data["seconds"] = round(time.monotonic() - started + report.data.get("seconds", 0))
            report.data["finished_at"] = now_iso()
            report.save()
            con.stop_heartbeat()
        verdicts = evaluate(report.data)
        report.data["verdicts"] = verdicts
        report.save()
        render_report(con, report.data, verdicts)
        con.line("")
        con.line(bi(f"报告已保存：{report.path}   求助时把这个文件发出来。", f"Report saved: {report.path}   Share this file when asking for help."))
        if con.tty and not args.no_sim:
            try:   # keep the 3D page alive until the user has looked at everything
                input(bi("回车退出（浏览器里的 3D 页面随之关闭）", "Enter to exit (the 3D page closes with it)") + " > ")
            except EOFError:
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
    report_path = WORK_DIR / f"report-{_dt.date.today().isoformat()}.json"
    report = Report(report_path, {"schema": 1, "tool_version": TOOL_VERSION, "lerobot_version": LEROBOT_VERSION,
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
    con.line(bi("全程 30–90 分钟，取决于网速与机器。中途 Ctrl-C 可停，已完成部分会保留。",
                "30-90 minutes depending on network and machine. Ctrl-C stops; finished parts are kept."))
    if not args.yes and con.tty:
        try:
            input(bi("回车继续，Ctrl-C 退出", "Enter to continue, Ctrl-C to quit") + " > ")
        except EOFError:
            pass

    # ---- install ------------------------------------------------------------------------------
    con.step(3, total_steps, f"安装 LeRobot {LEROBOT_VERSION}", f"installing LeRobot {LEROBOT_VERSION}", "5–20 min")
    if args.use_current_env:
        py = Path(sys.executable)
        probe = subprocess.run([str(py), "-c", IMPORT_PROBE], capture_output=True, text=True)
        if probe.returncode != 0:
            report.data["install"] = {"status": "FAIL", "reason": "current env cannot import lerobot/torch/viser", "stderr": probe.stderr[-2000:]}
            report.save()
            con.item("bad", "当前环境导入 lerobot / torch / viser 失败", "current env cannot import lerobot / torch / viser")
            return 1
        report.data["install"] = {"status": "PASS", "reused": True, **json.loads(probe.stdout.strip().splitlines()[-1])}
    elif args.skip_install and venv_python(WORK_DIR / ".venv").exists():
        py = venv_python(WORK_DIR / ".venv")
        probe = subprocess.run([str(py), "-c", IMPORT_PROBE], capture_output=True, text=True)
        report.data["install"] = {"status": "PASS" if probe.returncode == 0 else "FAIL", "reused": True,
                                  **(json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 else {"stderr": probe.stderr[-2000:]})}
    else:
        py = build_env(con, specs, report.data, args)
    report.save()
    inst = report.data["install"]
    if inst.get("status") != "PASS" or py is None:
        con.item("bad", f"LeRobot {LEROBOT_VERSION} 装不上：{inst.get('reason', '')}  日志 {inst.get('log', '')}",
                 f"LeRobot {LEROBOT_VERSION} did not install: {inst.get('reason', '')}  log {inst.get('log', '')}")
        report.data["seconds"] = round(time.monotonic() - started)
        report.save()
        con.stop_heartbeat()
        render_report(con, report.data, evaluate(report.data))
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
