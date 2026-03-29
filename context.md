Wrote findings to `/Users/mkohram/src/nanochat/context.md`.

Summary:
- `experiments/` is a standalone MQAR/GDH research sandbox.
- Main current entrypoints are:
  - `experiments/mqar_gdh_mps_lab.py`
  - active MPS launcher scripts in `experiments/`
  - live dashboard in `experiments/probe-dashboard/`
  - archived older GPU-oriented lab materials in `experiments/archive/gpu-lab/`
- It tests whether GDH can solve long-gap associative recall under forced sliding-window “blindfold” attention.
- It writes timestamped JSON/PNG artifacts to `experiments/out/` and optionally updates `probe_live.json` for the dashboard.
- I included structure, code snippets, architecture notes, and concrete commands/examples from checked logs.