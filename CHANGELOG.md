# Changelog

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
