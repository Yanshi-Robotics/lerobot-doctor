#!/usr/bin/env bash
# LeRobot Doctor for macOS: drag this file into a Terminal window and press Enter.
# (A zip from GitHub loses the executable bit, so double-clicking may not work; dragging does.)
cd "$(dirname "$0")"
if [ ! -f ./doctor.sh ]; then
  echo "旁边找不到 doctor.sh。请先把整个 ZIP 解压，再拖解压出来的文件夹里的 doctor.command。"
  echo "doctor.sh is not next to this file. Unzip the whole ZIP first, then drag doctor.command from the unzipped folder."
  exit 1
fi
exec bash ./doctor.sh "$@"
