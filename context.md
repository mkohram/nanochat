Wrote findings to `/Users/mkohram/src/nanochat/context.md`.

Summary:
- `experiments/` is a standalone MQAR/GDH research sandbox.
- Main current entrypoints are:
  - `experiments/mqar_gdh_lab.py`
  - `experiments/mqar_gdh_mps_lab.py`
  - launcher scripts in `experiments/run_probe_*.sh`
  - live dashboard in `experiments/probe-dashboard/`
- It tests whether GDH can solve long-gap associative recall under forced sliding-window “blindfold” attention.
- It writes timestamped JSON/PNG artifacts to `experiments/out/` and optionally updates `probe_live.json` for the dashboard.
- I included structure, code snippets, architecture notes, and concrete commands/examples from checked logs.