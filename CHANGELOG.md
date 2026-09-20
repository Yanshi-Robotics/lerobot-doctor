# Changelog

Versions are git tags `v<x.y.z>`; each heading links to its tag. A release moves four things in one
commit: `TOOL_VERSION` in `lerobot_doctor.py`, the default `DOCTOR_TAG` in `doctor.ps1` and
`doctor.sh`, and a section here; then `git tag -a v<x.y.z>` and push the tag before anyone pastes the
one-liner (`python3 tests/test_doctor.py` checks the first three agree). The two launcher file names
on `main` are a public API: every copy of the one-liner in the wild fetches them.

## [0.1.6](https://github.com/Yanshi-Robotics/lerobot-doctor/releases/tag/v0.1.6) - 2026-09-20

A full review after a real run on a CPU-only Windows laptop (i7-1065G7, 16 GB): the R2 route was right,
but several numbers and sentences around it were not. What changed, and why:

Measurements
- The simulated task no longer counts video decoding. Every frame is decoded before the clock starts;
  a real robot's cameras hand over frames for free, and on that laptop the pyav decoder (twice per
  step, as it turned out) had dragged a 3.06 s/chunk ACT down to "2.5 Hz". The demo now has three
  outcomes: PASS (kept up with the 30 Hz arm), SLOW (finished, but the arm stalls at each chunk
  refill: the refill wait is printed) and TOO_SLOW (cut off by the wall clock). The 3D page and the
  report print the same numbers. `time.perf_counter` throughout: `time.monotonic` ticks every
  15.6 ms on Windows, half a 30 Hz period.
- The realtime band is decided on the 95th percentile, not the median: the arm waits for the slowest
  chunk. The old p95 was computed wrong (the 80th percentile at n = 5) and never shown. Verified
  against both example reports: no stored verdict flips. A one-sample measurement (the early stop on a
  hopeless CPU) is marked `(n=1)`.
- A training probe that hits its budget keeps every timed step it finished: the projection says
  "~536.7 h · batch 8 -> cloud (projected from 2 steps; the run stopped at its 15 min budget)"
  instead of "not tested: very slow disk or network". A timeout with no step names the phase it was in.
- On a CPU or MPS machine, once a smaller level already needs the cloud, the larger levels skip their
  training probe (their inference is still measured): that was ~45 minutes of guaranteed timeouts.
- A Windows access violation (0xC0000005) is a tool/environment crash, not "out of RAM": it no longer
  makes every larger level skip or the route say R4.
- ACT's measurement is reported before the bonus after-training demo starts, so an overrunning demo
  cannot erase it; a demo with fewer than 30 extra steps is skipped and says so; the report prints the
  before/after joint error and the batch ladder (8 OOM -> 4 OK, peak GB).

Robustness
- A probe's deadline is checked every 50 ms whether or not the child prints: a child hung in silence
  used to hang the tool forever. On POSIX the child runs in its own session and the whole tree is killed;
  the pipe is closed and the process reaped; the last unterminated line still reaches the parser.
  `uv venv` and `uv pip install` have timeouts (10 min / 60 min per attempt, the install retries from cache).
- Every run writes its own `report-<date>-<hhmm>.json` plus `report-latest.json`; a rerun on the same
  day no longer overwrites the previous report. Writes are atomic and fsynced; a stray Path or numpy
  scalar can no longer lose a report.
- A retry is classified by its own log output, not the previous attempt's; the parent's bookkeeping
  (`evidence`, `seconds`) wins over anything the worker wrote under the same key; a `@@progress` line
  cut by the pipe no longer crashes the parent; worker notes (the "40x over budget, stopping early"
  line among them) were sent on a channel the parent did not read and now reach the screen.
- Two `None <= 0` comparisons in the prechecks (a worker that OOMs before counting its parameters)
  no longer crash the whole run. The viser page is guarded at every call: a page error closes the page,
  the probes continue; the real port is printed (and a busy port says `--port`); the server is stopped
  at exit; the "Enter to exit" prompt appears only when the page exists.
- Mesh downloads are atomic (`.part` then rename): a truncated STL was kept forever. `--skip-install`
  checks the venv's lerobot version and the report says when it differs; a venv with the wrong Python
  is rebuilt. `--uninstall` runs inside the crash handler. Import probes have a timeout and read the
  last JSON line (torch may print after it). Worker stdout is `utf-8:replace`; Hugging Face download
  timeouts are raised from 10 s to 60 s. The token is sent to huggingface.co only, never to a mirror.
  On Windows a legacy console is switched to UTF-8 instead of printing the Chinese half as `?`.

Wording (only what was measured)
- An integrated GPU is "a non-NVIDIA GPU (integrated ones included)", not a discrete one, and the
  final report says the machine has it and PyTorch does not use it. "30 to 90 minutes" became
  "30-60 minutes on a GPU; 1 to 2.5 hours CPU-only". The basics line says assemble / calibrate /
  teleoperate / record are not tested by this tool. A dataset failure names its cause instead of
  "network". uv failing to build the environment is reported as uv's problem, with the mirror hint,
  not as "LeRobot did not install". R2 also names the highest level whose inference runs locally.
  A batch-1 training verdict is a yellow cell, never green. Ctrl-C prints the report path.

Launchers
- `uv python find 3.12` runs first; the GitHub download of a standalone Python happens only when the
  machine has none, and its failure is an error naming `UV_PYTHON_INSTALL_MIRROR` (the ps1 ignored the
  exit code; the sh died silently under `set -e`). The sh has an error trap like the ps1's catch block.
- Under `curl | bash` the sh no longer falls back to `$0`: a stray `lerobot_doctor.py` in the current
  folder used to run instead of the release. `doctor.bat` keeps the ps1's exit code; `doctor.command`
  checks for `doctor.sh`. `$LASTEXITCODE` is guarded; downloads have a timeout and no per-byte
  progress bar; `UV_INSTALL_DIR` is honoured.
- The report records which launcher and tag ran it and which Python; `--version` prints them.

Tests: 63 -> 127 (the worker protocol parser, the process runner with a fake child, the demo and p95
arithmetic, every status through both verdicts, the speed rule, the launchers driven with a fake uv).
One test had been unreachable since 0.1.4 (indented inside a helper); it runs now.

## 0.1.5 - 2026-09-20

- The final report is printed twice: a complete English box first, then a complete Chinese one.
  Mixed-language cells made both halves hard to scan. The JSON report, the launchers and the Quick
  start command are unchanged.

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
