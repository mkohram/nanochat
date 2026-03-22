# AGENTS

## Style
- Keep answers short and concise.
- No emojis in commits, issues, PR comments, or code.
- No fluff or cheerful filler text.
- Technical prose only; be kind but direct.
- Prefer: "Thanks @user".
- Avoid: "Thanks so much @user!".

## Experiments directory working context

Primary workstream is `experiments/`.

### What it is
- `experiments/` is a self-contained GDH/MQAR research sandbox, separate from mainline production training.
- Main purpose: test whether GDH sidecar memory can solve long-gap associative recall when direct attention is intentionally blindfolded.
- Treat code in `experiments/` as the local source of truth for this workstream. `SPEC.md` and `GDH_SHARED_UNDERSTANDING.md` are useful orientation docs but may lag behind current experiment code.

### Core task being studied
- The main task is blindfolded MQAR (multi-query associative recall).
- Sequences are constructed as:
  - key/value pairs
  - filler gap
  - query/answer pairs
- Training/eval only score answer positions.
- Blindfolding is enforced by forcing sliding-window attention (`swa_window`) on all layers and requiring `gap_min > swa_window`, so long-gap retrieval must come from GDH sidecar memory rather than direct attention.

### Main entrypoints
- `experiments/mqar_gdh_lab.py`
  - baseline CPU/CUDA-oriented minimal lab harness.
- `experiments/mqar_gdh_mps_lab.py`
  - MPS-oriented variant with device selection, autocast policy, optional `torch.compile`, wall-time logging, and throttled expensive metrics.
- Launcher scripts:
  - `experiments/run_probe_easy.sh`
  - `experiments/run_probe_hard.sh`
  - `experiments/run_probe_easy_mps.sh`
  - `experiments/run_probe_hard_mps.sh`
- Dashboard:
  - `experiments/run_probe_dashboard.sh`
  - `experiments/probe-dashboard/`

### Reduced lab architecture
The lab harness is intentionally narrower than canonical GDH. It is a first-principles bench for debugging.

Per layer, the reduced GDH path is effectively:
1. token embeddings + norm
2. GDH read from sidecar into local stream
3. standard transformer block
4. GDH write proposal into slots
5. tokenwise sidecar accumulation over sequence

Important knobs exposed by the lab harnesses:
- `--arch {baseline,gdh}`
- `--betas` for additive vs leaky scan behavior
- `--route-topk` for sparse routing
- `--usage-balance-lambda` for slot-usage regularization
- `--gdh-slots`
- `--gdh-write-heads`
- `--swa-window`
- MQAR geometry knobs such as `--n-pairs`, `--n-queries`, `--gap-min`, `--gap-max`

### Canonical configs already encoded in launcher scripts
Easy blindfold probe:
- short sequence
- `sequence_len=64`
- `n_layer=3`, `n_head=4`, `n_embd=64`
- `gdh_slots=8`
- `route_topk=4`
- `usage_balance_lambda=0.01`
- `swa_window=8`
- `n_pairs=2`, `n_queries=2`
- `gap_min=16`, `gap_max=32`
- `vocab_size=128`

Hard blindfold probe:
- longer sequence and larger gap
- `sequence_len=256`
- `n_layer=4`, `n_head=8`, `n_embd=128`
- `gdh_slots=8`
- `route_topk=4`
- `usage_balance_lambda=0.01`
- `swa_window=8`
- `n_pairs=16`, `n_queries=8`
- `gap_min=64`, `gap_max=192`
- `vocab_size=128`
- hard config also enforces capacity stress (`n_pairs > gdh_slots`)

### Outputs and artifacts
Experiment outputs go under `experiments/out/`:
- timestamped JSON summaries
- PNG plots
- launcher logs
- `probe_live.json` for the live dashboard
- `probe_current_log_path.txt` pointing at the active launcher log
- profiling outputs in `experiments/out/profile_*`

Typical lab harness outputs:
- `mqar_gdh_lab_<timestamp>.json/.png`
- `mqar_gdh_mps_lab_<timestamp>.json/.png`

### Dashboard behavior
- `experiments/probe-dashboard/` is a React/Vite viewer-only dashboard.
- It polls a JSON source, normally `experiments/out/probe_live.json`.
- It visualizes:
  - eval top-1 / MRR
  - wall time vs step
  - train vs eval CE
  - layerwise slot-collapse metrics
  - out-state histograms
  - sidecar heatmaps
- It does not run experiments itself; probe scripts write the JSON, dashboard reads it.

### Existing logs/docs worth checking before changes
- `context.md`
- `experiments/EXPERIMENT_LOG.md`
- `experiments/PROBE_V2_LOG.md`
- `SPEC.md`
- `GDH_SHARED_UNDERSTANDING.md`

### Known caution points
- Some historical docs describe broader or older GDH behavior than the current minimal lab harness.
- Verify launcher/script arg compatibility before relying on a launcher unchanged.
- In particular, check `experiments/run_probe_hard.sh` against current argparse in `experiments/mqar_gdh_lab.py` before using it, because there may be stale flags.
