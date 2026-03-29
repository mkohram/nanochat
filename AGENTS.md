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
- `experiments/mqar_gdh_mps_lab.py`
  - current primary minimal lab harness.
  - includes device selection, autocast policy, optional `torch.compile`, wall-time logging, and throttled expensive metrics.
- Launcher scripts:
  - `experiments/run_probe_easy_mps.sh`
  - `experiments/run_probe_hard_mps.sh`
- Dashboard:
  - `experiments/run_probe_dashboard.sh`
  - `experiments/probe-dashboard/`
- Archived older GPU-oriented lab and runners live in:
  - `experiments/archive/gpu-lab/`

### Reduced lab architecture
The lab harness is intentionally narrower than canonical GDH. It is a first-principles bench for debugging.

Per layer, the active reduced GDH path is effectively:
1. token embeddings + norm
2. residual blend with the input stream (`x` / `x0` lambdas)
3. GDH read from sidecar into local stream
4. standard transformer block
5. GDH write proposal into slots
6. tokenwise sidecar accumulation over sequence

The write path is probe-local and currently supports routing ablations:
- `static`: learned slot addresses only
- `content`: current sidecar contents only
- `hybrid`: static + content logits

### Important knobs
Current active knobs in `experiments/mqar_gdh_mps_lab.py`:
- `--arch {baseline,gdh}`
- `--betas` for additive vs leaky scan behavior
- `--route-topk` for sparse routing
- `--write-routing {static,content,hybrid}` for write-routing ablations
- `--gdh-slots`
- `--gdh-write-heads`
- `--swa-window`
- MQAR geometry knobs such as `--n-pairs`, `--n-queries`, `--gap-min`, `--gap-max`

MPS-only or MPS-primary knobs in `experiments/mqar_gdh_mps_lab.py`:
- `--gdh-use-write-brain`
- `--gdh-write-brain-hidden-mult`
- `--read-mute-gate`
- `--amp-dtype`
- `--compile`
- `--full-metrics-every`

Current active probe line does not use usage-balance loss.

### Canonical configs already encoded in launcher scripts
Easy blindfold probe:
- short sequence
- `sequence_len=64`
- `n_layer=3`, `n_head=4`, `n_embd=64`
- `gdh_slots=8`
- `gdh_write_heads=1`
- `route_topk=4`
- `write_routing=static`
- `swa_window=8`
- `n_pairs=2`, `n_queries=2`
- `gap_min=16`, `gap_max=32`
- `vocab_size=128`

Hard blindfold probe:
- longer sequence and larger gap
- `sequence_len=256`
- `n_layer=4`, `n_head=8`, `n_embd=128`
- `gdh_slots=8`
- `gdh_write_heads=1`
- `route_topk=4`
- `write_routing=static`
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
- launcher logs such as `probe_easy_mps_<steps>_<timestamp>.log`

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
- Current collapse telemetry includes mean, max, and p90 off-diagonal slot cosine views.
- It does not run experiments itself; probe scripts write the JSON, dashboard reads it.

### Existing logs/docs worth checking before changes
- `context.md`
- `experiments/PROBE_V3_LOG.md`
- archived legacy scan-probe materials in `experiments/archive/scan-probe/`
- `SPEC.md`
- `GDH_SHARED_UNDERSTANDING.md`

### Known caution points
- Some historical docs describe broader or older GDH behavior than the current minimal lab harness.
- Verify launcher/script arg compatibility before relying on a launcher unchanged.
- Treat `experiments/mqar_gdh_mps_lab.py` plus the canonical launcher scripts as the most current probe surface.
- Archived older GPU-oriented lab material lives in `experiments/archive/gpu-lab/`; treat it as legacy unless explicitly needed.
