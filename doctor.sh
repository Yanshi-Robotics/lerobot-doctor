#!/usr/bin/env bash
# LeRobot Doctor launcher for macOS and Linux.
#   curl -LsSf https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/main/doctor.sh | bash
# or, from an unpacked zip:   bash doctor.sh
#
# It does three things and nothing else: make sure `uv` exists, make sure Python 3.12 exists,
# run lerobot_doctor.py. Everything after that is the Python file's job.
set -euo pipefail

DOCTOR_TAG="${DOCTOR_TAG:-v0.1.6}"   # must equal the tag this file is published under
export DOCTOR_TAG                     # the program records which tag ran it
RAW="https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/${DOCTOR_TAG}"
ISSUES="https://github.com/Yanshi-Robotics/lerobot-doctor/issues/new"
HOME_DIR="${HOME}/lerobot-doctor"
UV_HOME="${UV_INSTALL_DIR:-${HOME}/.local/bin}"   # where uv's installer puts the binary

say() { printf '%s\n' "$*"; }

# `set -e` alone dies in silence. Mirrors doctor.ps1's catch block: the step, the exit code, a hint
# when one applies, and where to report it. (`exec` at the end drops the trap, as it should: from
# there on the Python program handles its own errors.)
step="start"
hint=""
on_err() {
  rc=$?
  say ""
  say "启动器出错了，体检没有开始 | The launcher failed; the check did not start"
  say "  卡在 | step : ${step}"
  say "  退出码 | exit code: ${rc}"
  [ -n "${hint}" ] && say "  ${hint}"
  say "  最常见是网络问题：重试一次；下载慢可以把 HF_ENDPOINT 设成镜像 | Most often a network problem: try again; if downloads are slow, point HF_ENDPOINT at a mirror"
  say "  报 issue 请把这个窗口截图贴上去 | to report it, paste a screenshot of this window at: ${ISSUES}"
}
trap on_err ERR
mkdir -p "${HOME_DIR}"

# 1. the script itself: next to this launcher (zip) or downloaded (one-liner). Under `curl | bash`
#    BASH_SOURCE is unset and $0 is "bash": never fall back to $0 or the current folder, or a stray
#    lerobot_doctor.py lying there would run instead of the release.
step="download lerobot_doctor.py (${DOCTOR_TAG})"
here=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "${here}" ] && [ -f "${here}/lerobot_doctor.py" ]; then
  SCRIPT="${here}/lerobot_doctor.py"
else
  SCRIPT="${HOME_DIR}/lerobot_doctor.py"
  say "下载体检程序 | downloading lerobot_doctor.py (${DOCTOR_TAG})"
  curl -LsSf "${RAW}/lerobot_doctor.py" -o "${SCRIPT}"
fi

# 2. uv (single static binary)
step="install uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "${UV_HOME}/uv" ]; then
  say "安装 uv（Python 环境管理器，约 30 MB） | installing uv (~30 MB)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${UV_HOME}:${PATH}"

# 3. Python 3.12: one uv already knows (system or managed) is enough; the GitHub download of a
#    standalone build runs only when there is none.
step="find Python 3.12"
say "准备 Python 3.12 | preparing Python 3.12"
if py="$(uv python find --no-project 3.12 2>/dev/null)"; then
  say "  已有 Python 3.12：${py} | using Python 3.12 at ${py}"
else
  step="install Python 3.12"
  hint="下载 Python 3.12 失败；防火墙后请先 export UV_PYTHON_INSTALL_MIRROR=<镜像> 再重跑 | the Python 3.12 download failed; behind a firewall export UV_PYTHON_INSTALL_MIRROR=<mirror> and rerun"
  say "  下载 Python 3.12（约 30 MB） | downloading Python 3.12 (~30 MB)"
  uv python install 3.12
  hint=""
fi

# 4. run. Stage 1 needs only the stdlib, so `uv run` with no dependencies is enough;
#    the script builds its own venv for stage 2.
step="run"
export PYTHONIOENCODING=utf-8
export DOCTOR_LAUNCHER=sh   # tells the program a launcher is around it
exec uv run --python 3.12 --no-project "${SCRIPT}" "$@"
