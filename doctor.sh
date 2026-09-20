#!/usr/bin/env bash
# LeRobot Doctor launcher for macOS and Linux.
#   curl -LsSf https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/main/doctor.sh | bash
# or, from an unpacked zip:   bash doctor.sh
#
# It does three things and nothing else: make sure `uv` exists, make sure Python 3.12 exists,
# run lerobot_doctor.py. Everything after that is the Python file's job.
set -euo pipefail

DOCTOR_TAG="${DOCTOR_TAG:-v0.1.5}"   # must equal the tag this file is published under
RAW="https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/${DOCTOR_TAG}"
HOME_DIR="${HOME}/lerobot-doctor"
mkdir -p "${HOME_DIR}"

say() { printf '%s\n' "$*"; }

# 1. the script itself: next to this launcher (zip) or downloaded (one-liner)
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
if [ -n "${here}" ] && [ -f "${here}/lerobot_doctor.py" ]; then
  SCRIPT="${here}/lerobot_doctor.py"
else
  SCRIPT="${HOME_DIR}/lerobot_doctor.py"
  say "下载体检程序 | downloading lerobot_doctor.py (${DOCTOR_TAG})"
  curl -LsSf "${RAW}/lerobot_doctor.py" -o "${SCRIPT}"
fi

# 2. uv (single static binary; installs into ~/.local/bin)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "${HOME}/.local/bin/uv" ]; then
  say "安装 uv（Python 环境管理器，约 30 MB） | installing uv (~30 MB)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

# 3. Python 3.12 (uv fetches a standalone build if the system has none)
say "准备 Python 3.12 | preparing Python 3.12"
uv python install 3.12 --quiet

# 4. run. Stage 1 needs only the stdlib, so `uv run` with no dependencies is enough;
#    the script builds its own venv for stage 2.
export PYTHONIOENCODING=utf-8
export DOCTOR_LAUNCHER=sh   # tells the program a launcher is around it
exec uv run --python 3.12 --no-project "${SCRIPT}" "$@"
