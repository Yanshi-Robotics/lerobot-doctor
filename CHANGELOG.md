# Changelog

## 0.1.4 - 2026-09-19

Windows: a tester saw the window vanish ("flash crash"). Root causes and the fixes:

- `doctor.ps1` ended with `exit $LASTEXITCODE`. Under `irm ... | iex` that exits the user's own
  PowerShell session, so any early return (a hard floor, a failed install, a crash) closed the window
  with everything on it. The work now runs in a function that returns a code; a file-mode start
  (doctor.bat, right-click "Run with PowerShell") waits for Enter before the window closes. The uv
  installer runs in a child PowerShell for the same reason (it calls `exit 1` on failure).
- A crash of the program itself is now caught at the top level: it writes
  `~/lerobot-doctor/logs/crash-<time>.log` (platform, Python, arguments, environment, log listing,
  full traceback), prints the error, where it happened, the log path and the issue URL, and on a
  Windows console with no launcher around it waits for Enter. Exit code 70.
- `doctor.ps1` is pure ASCII: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page
  (Chinese literals became mojibake), and a BOM breaks `irm | iex`. Chinese messages are `\u` escapes
  checked by a test. `$ErrorActionPreference` no longer leaks into the user's session; TLS 1.2 is enabled
  for old Windows 10 builds.
- `doctor.bat` explains what to do when run from inside the ZIP (doctor.ps1 missing), and is CRLF.
- Console output can no longer die on a character the terminal cannot encode (`errors="replace"`).
- Windows 11 is reported as Windows 11 (build >= 22000), not "Windows 10 (10.0.22xxx)".
- A crashed or timed-out probe names its error line and its log file in the verdict.
- The Quick start one-liners fetch the launcher from `main`, a URL that never changes (the way uv's
  and rustup's installers work); the launcher pins the program to the release tag it was published
  with. Re-running the same command after a release now gets the new release. Commands copied from
  the 0.1.3 README still point at the `v0.1.3` tag and keep running 0.1.3; copy the command once more.

## 0.1.3 - 2026-09-19

- The final report is a boxed, structured card in the terminal: machine, basics, a per-level
  table (inference / training), details with the evidence behind every non-green cell, and
  the SO-101 route. Fitted to the terminal width, colours on a terminal, plain glyphs in a log.
  The JSON file is still written and its path is printed under the box.

## 0.1.2 - 2026-09-19

- The launchers' default `DOCTOR_TAG` still said `v0.1.0`, so the `v0.1.1` one-liner fetched the
  0.1.0 program. The default now equals the published tag; a release check greps for it.

## 0.1.1 - 2026-09-19

- Progress display: one status line redrawn in place, cut to the terminal width (a wrapped line
  left a copy of itself on every redraw); known-length work shows a conda-style bar; log files
  get milestones only (bar quarters, one heartbeat per 30 s).
- Shorter activity labels.

## 0.1.0 - 2026-09-19

- First release: spec check, private `uv` environment with `lerobot==0.6.1`, five-level ladder
  (ACT, Diffusion, SmolVLA, X-VLA, WALL-OSS) with real inference, a 3D SO-101 task in the
  browser and real training steps; evidence-based verdicts; bilingual terminal output and JSON report.
