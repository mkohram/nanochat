# PROBE_V2_LOG.md

## Purpose
Fresh experiment log for the new self-contained probe track.

- Script: `experiments/mqar_scan_beta_probe_v2.py`
- Origin: copied from `experiments/mqar_scan_beta_probe.py`
- Date created: 2026-02-27
- Goal: reproduce prior probe behavior first, then iterate only inside this testbed before touching mainline training code.

---

## Phase 0 — Repro baseline (must pass before changes)

Status: TODO

### Repro checklist
- [x] Run the copied probe with known-good settings from previous probe phase
- [x] Confirm output JSON/PNG generation in `experiments/out/`
- [x] Confirm key metrics are in expected range (acc / mrr / slot cosine trends)
- [x] Record exact command and artifacts

### Command(s)
1) Baseline blindfold replication:
- `.venv/bin/python experiments/mqar_scan_beta_probe_v2.py --arch baseline --swa-window 8 --steps 300 --log-every 50 --sequence-len 256 --n-layer 4 --n-head 8 --n-embd 128 --n-pairs 16 --n-queries 8 --gap-min 64 --gap-max 192 --batch-size 8 --eval-batch-size 8 --seed 123`

2) GDH blindfold replication:
- `.venv/bin/python experiments/mqar_scan_beta_probe_v2.py --arch gdh --swa-window 8 --steps 300 --log-every 50 --sequence-len 256 --n-layer 4 --n-head 8 --n-embd 128 --gdh-slots 8 --gdh-write-heads 8 --route-topk 4 --usage-balance-lambda 0.01 --n-pairs 16 --n-queries 8 --gap-min 64 --gap-max 192 --batch-size 8 --eval-batch-size 8 --seed 123`

### Results
- Baseline artifacts:
  - JSON: `experiments/out/mqar_scan_beta_probe_20260227_143526.json`
  - PNG: `experiments/out/mqar_scan_beta_probe_20260227_143526.png`
  - Final: `eval_acc_top1=0.0`, `eval_mrr=0.00294`, `eval_ce=7.9772`

- GDH artifacts:
  - JSON: `experiments/out/mqar_scan_beta_probe_20260227_143847.json`
  - PNG: `experiments/out/mqar_scan_beta_probe_20260227_143847.png`
  - Final across beta sweep (`1.0,0.99,0.95,0.9`):
    - `eval_acc_top1=0.0` for all
    - `eval_mrr~0.00265-0.00281`
    - `eval_ce~7.98`
    - `slot_cos_last~0.992-0.996` (severe collapse)

- Interpretation:
  - Blindfolding worked (baseline fails), but GDH also fails under this exact copied setup on current code.
  - Probe v2 replication now gives a clear failure baseline to improve from.

---

## Change policy for this track
- Keep `mqar_scan_beta_probe_v2.py` self-contained.
- Make one architectural change at a time.
- Log every change + command + result in this file.
- Do **not** promote to mainline until probe evidence is clear.

---

## Experiment entries

### 2026-02-27 — Entry 01 (Repro boot + blindfold check)
- Date/time: 2026-02-27 14:35–14:39
- Hypothesis: copied probe can reproduce blindfold setup behavior and provide a stable testbed baseline.
- Change:
  - Compatibility fix in `mqar_scan_beta_probe_v2.py`: removed deprecated `gdh_use_read_gate` arg passed to `GPTConfig` (no longer accepted in current codebase).
- Command:
  - see Phase-0 commands above (baseline + GDH blindfold runs).
- Key outputs:
  - baseline run file: `mqar_scan_beta_probe_20260227_143526.{json,png}`
  - gdh run file: `mqar_scan_beta_probe_20260227_143847.{json,png}`
  - both baseline and gdh failed top1 under these settings; GDH slot cosine collapsed to ~0.99.
- Verdict:
  - Probe is operational and reproducible as a self-contained harness.
  - Current copied configuration is a valid failure baseline (useful for controlled stabilization work).
- Next:
  - define first controlled ablation inside probe v2 (single change at a time), starting with write/update dynamics and slot-collapse pressure.

### 2026-02-27 — Entry 02 (legacy-freeze reconstruction attempt)
- Date/time: 2026-02-27 19:35–19:40
- Hypothesis: failure to reproduce may come from drift in imported `nanochat.gpt` GDH defaults; force probe-era internals explicitly.
- Change:
  - Created `experiments/mqar_scan_beta_probe_frozen_20260223.py`.
  - Added `_apply_legacy_20260223_gdh_freeze()` to force probe-era-style GDH init/gating assumptions:
    - disable read-mute gate in probe run,
    - reinit GDH matrices to legacy normal init style,
    - keep scalar probe write gate injection (`D->1`, bias=-2.0),
    - warmstart first two `W_o_read`.
- Command:
  - `.venv/bin/python experiments/mqar_scan_beta_probe_frozen_20260223.py --arch gdh --betas 1.0 --steps 2000 --log-every 500 --seed 123 --sequence-len 128 --n-layer 3 --n-head 4 --n-embd 64 --gdh-slots 16 --gdh-write-heads 8 --route-topk 2 --usage-balance-lambda 0.01 --swa-window 8 --n-pairs 4 --n-queries 4 --gap-min 16 --gap-max 32 --batch-size 32 --eval-batch-size 8 --eval-topk 5 --lr 3e-4`
- Key outputs:
  - JSON: `experiments/out/mqar_scan_beta_probe_20260227_194021.json`
  - PNG: `experiments/out/mqar_scan_beta_probe_20260227_194021.png`
  - Final: `eval_acc_top1=0.0`, `eval_mrr=0.00169`, `slot_cos_last=0.9922`
- Verdict:
  - Still collapses; legacy-like patching inside current code is insufficient.
- Next:
  - hard-freeze full probe stack by pinning exact historical model components (not just script-level patching).

### Entry template
- Date/time:
- Hypothesis:
- Change:
- Command:
- Key outputs:
- Verdict:
- Next:
