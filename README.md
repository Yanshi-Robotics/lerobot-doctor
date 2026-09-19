# LeRobot Doctor

**这台电脑能不能跑 LeRobot 0.6.1 做 SO-101 的具身智能任务？** 一条命令，真装、真跑、真测，给出结论。
**Can this computer run LeRobot 0.6.1 for an SO-101 arm?** One command; it installs, runs and measures for real, then tells you.

## 怎么用 · How to use

| 系统 System | 打开 Open | 粘贴这一行 Paste this line |
|---|---|---|
| macOS | 终端 Terminal | `curl -LsSf https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/v0.1.0/doctor.sh \| bash` |
| Linux | 终端 Terminal | 同上 same as above |
| Windows 10/11 | PowerShell | `irm https://raw.githubusercontent.com/Yanshi-Robotics/lerobot-doctor/v0.1.0/doctor.ps1 \| iex` |

不想粘命令：Download ZIP → 解压 → Windows 双击 `doctor.bat`；macOS 把 `doctor.command` 拖进终端回车；Linux `bash doctor.sh`。
Prefer a zip: Download ZIP → unpack → Windows: double-click `doctor.bat`; macOS: drag `doctor.command` into Terminal; Linux: `bash doctor.sh`.

全程 30–90 分钟（取决于网速），需要约 30 GB 磁盘。屏幕上一直有进度；中途 `Ctrl-C` 可停，已完成的部分保留。
Takes 30–90 minutes (mostly downloads) and ~30 GB of disk. The screen always shows progress; `Ctrl-C` stops and keeps what finished.

## 它做什么 · What it does

1. **读规格** — 系统、Python、显卡、内存、磁盘。只有「远低于要求」的硬门槛（磁盘 < 30 GB、内存 < 8 GB、Windows < 10）会在这里停。
   **Specs** — OS, Python, GPU, RAM, disk. Only hard floors far below the requirement stop here (disk < 30 GB, RAM < 8 GB, Windows < 10).
2. **真装 LeRobot 0.6.1** — 在 `~/lerobot-doctor/.venv` 里，装不上就是结论。
   **Installs LeRobot 0.6.1** for real in `~/lerobot-doctor/.venv`; a failed install is itself the verdict.
3. **五级阶梯，每级真跑** — ACT → Diffusion → SmolVLA → π0-FAST → π0.5。每级：下载权重 → 用官方 SO-101 样例录像做输入，量一次前向要多久 → 在浏览器里的 SO-101 3D 模拟臂上执行 10 秒任务 → 再真训几步，量每步多久、显存多少，batch 从大往小试。
   **Five levels, each run for real** — ACT → Diffusion → SmolVLA → π0-FAST → π0.5. Per level: download weights → time a forward pass on the official SO-101 sample recording → drive a 3D SO-101 in your browser for a 10-second task → train a few real steps, measuring step time and memory, batch size stepping down.
4. **结论** — 每级「本地实时推理 / 本地训练约 N 小时 / 需要上云」，再合成一句 SO-101 路线。每个 ⛔ 都带证据：实测失败，或一条能算出来的硬门槛（如「权重 7.2 GB > 显存 4 GB」）。其它一律写「未测 + 原因」。
   **Verdicts** — per level: real-time local inference / local training ≈ N hours / needs cloud, then one SO-101 route. Every ⛔ carries evidence: a measurement or an arithmetic floor (e.g. "weights 7.2 GB > VRAM 4 GB"). Everything else says "not tested" with the reason.

报告同时落在 `~/lerobot-doctor/report-<date>.json`，求助时把它发出来。
The report is also saved to `~/lerobot-doctor/report-<date>.json`; share that file when asking for help.

## 常见问题 · FAQ

- **π0-FAST / π0.5 显示「未测：许可」** — 它们的 tokenizer 来自 `google/paligemma-3b-pt-224`，需要在 Hugging Face 上同意许可并登录。体检会问你要 token；直接回车就跳过这两级。
  **π0-FAST / π0.5 say "not tested: license"** — their tokenizer comes from `google/paligemma-3b-pt-224`, gated on Hugging Face. The check asks for a token; press Enter to skip those two levels.
- **中国大陆网络** — 运行前 `export HF_ENDPOINT=https://hf-mirror.com`（PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。
  **Slow access to Hugging Face** — set `HF_ENDPOINT` to a mirror before running.
- **卸载** — `python ~/lerobot-doctor/lerobot_doctor.py --uninstall`（删 `~/lerobot-doctor`，保留 Hugging Face 缓存，将来上课直接复用）。
  **Uninstall** — same command with `--uninstall`; removes `~/lerobot-doctor`, keeps the Hugging Face cache for later use.
- 更多开关：`--specs-only`、`--skip-pi`、`--device cpu`、`--no-sim`、`--port`。
  More switches: `--specs-only`, `--skip-pi`, `--device cpu`, `--no-sim`, `--port`.

## 开发 · Development

```bash
python tests/test_doctor.py          # rules, ladder state machine, parsers; no network, no torch
python lerobot_doctor.py --specs-only
```

`lerobot_doctor.py` 是唯一的程序文件；启动器只负责装 `uv` 和 Python 3.12。阈值全部是带来源注释的具名常量。
`lerobot_doctor.py` is the only program file; the launchers only install `uv` and Python 3.12. Every threshold is a named constant with its source.

MIT License.
