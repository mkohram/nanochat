# MQAR Experiment Log

## 2026-02-27: GDH V2.2 plan (forced-hearing pass)

### Versioning
- New experiment line designated **GDH v2.2**.
- Motivation: v2.1 showed read-mute collapse in long run telemetry (`read_mute_open_frac ~ 0`), causing a “deaf scribe” failure mode.

### Planned changes from v2.1 -> v2.2
1. **Restore global/sidecar LR to full speed**
   - Remove the 0.25x multiplier on GDH global/tied param group.
   - Goal: prevent lagging sidecar representation learning relative to backbone.

2. **Leaky read-mute gate floor**
   - Replace:
     - `g_read_mute = sigmoid(logit)`
   - With:
     - `g_read_mute = 0.10 + 0.90 * sigmoid(logit)`
   - Keep existing init values unless explicitly changed in ablation.
   - Goal: preserve non-zero read path and avoid gradient starvation through full mute collapse.

### Acceptance checks for v2.2
- `gdh/read_mute_open_frac` should stay materially above zero through training.
- Overall `train/loss` gap vs baseline should improve relative to v2.1.
- Preserve or improve chunk-late metrics (`train/chunk_loss_avg_9`, `train/chunk_loss_gap_vs_0`).

### Implementation status
- Code changes applied:
  - read-mute gate updated to `0.10 + 0.90 * sigmoid(logit)` in `nanochat/double_helix.py`.
  - read-mute telemetry probe updated to the same formula in `scripts/local_base_train.py`.
  - GDH global/tied optimizer LR multiplier restored from `0.25` to `1.0` in `nanochat/gpt.py`.
  - GDH oracle + stability tests updated for v2.2 behavior.
- Validation:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/gdh/test_stability_controls.py tests/gdh/test_oracle.py`
  - Result: `60 passed`.
- Training launch (initial, later stopped due to unintended eval at step 250):
  - Run name: `gdh-v22-forced-hearing-recurrent10-s256-b256-110k-20260227-r2`
  - W&B run: `mkohram-none/nanochat/rt0dmzhd`
  - Log: `runlogs/gdh_v22_forced_hearing_recurrent10_s256_b256_110k_20260227.log`
- Training relaunch (no-eval policy):
  - Run name: `gdh-v22-forced-hearing-recurrent10-s256-b256-110k-noeval-20260227`
  - W&B run: `mkohram-none/nanochat/3dvn2xo9`
  - Log: `runlogs/gdh_v22_forced_hearing_recurrent10_s256_b256_110k_noeval_20260227.log`
  - Full command:
    - `.venv/bin/python -m scripts.local_base_train --run gdh-v22-forced-hearing-recurrent10-s256-b256-110k-noeval-20260227 --arch gdh --depth 4 --aspect-ratio 96 --head-dim 32 --max-seq-len 256 --device-batch-size 1 --total-batch-size 256 --num-iterations 110000 --dataset fineweb_edu --loader-mode recurrent_doc --recurrent-max-chunks-per-doc 10 --gdh-recurrent-carry-state --gdh-slots 20 --gdh-write-heads 12 --gdh-use-write-gate --gdh-write-gate-bias 1.0 --gdh-write-brain-hidden-mult 1 --no-value-embeds --eval-every -1 --core-metric-every -1 --sample-every -1 --save-every -1`
- Guardrail for future runs:
  - For this experiment family, always explicitly pass `--eval-every -1` (and keep `--core-metric-every -1`) unless Mojtaba asks for eval.
  - Launch long runs detached with `nohup ... &` and redirect to a runlog file; do not rely on foreground `exec` runs for multi-hour training (they can be reaped/interrupted with session lifecycle).

## 2026-02-27: GDH V2.4 launch (write-whisper pass)

### Why
- v2.3 telemetry still showed read-mute collapse trend over time.
- Auditor diagnosis: `delta = tanh(RMSNorm(delta_raw))` can bias write updates toward same-sign magnitude under positive-skewed activations.

### v2.4 change (single surgical ablation)
- Removed pre-tanh RMSNorm from write update path.
- New write bound path is strictly:
  - `delta = tanh(delta_raw)`
- Kept all other v2.3 knobs locked:
  - read-mute floor: `0.05 + 0.95*sigmoid(...)`
  - sidecar/global LR multiplier: `1.0x`
  - `W_o_read = 0` ReZero output coupling
  - uniform init parity for GDH matrices
  - `W_o_write` active (non-zero init)

### Code touch points
- `nanochat/gpt.py`
  - removed `rms_norm(delta)` before `tanh(delta)` in main GDH forward.
- `scripts/local_base_train.py`
  - telemetry probe path updated to mirror mainline (`delta = tanh(delta)`).

### Validation
- `PYTHONPATH=. .venv/bin/pytest -q tests/gdh/test_stability_controls.py tests/gdh/test_oracle.py`
- Result: `60 passed`.

### Run control
- Stopped prior v2.3 run.
- Launched v2.4 (nohup, no-eval):
  - Run name: `gdh-v24-no-rmsnorm-delta-tanh-recurrent10-s256-b256-110k-noeval-20260227`
  - W&B run: `mkohram-none/nanochat/4271ih41`
  - Log: `runlogs/gdh_v24_no_rmsnorm_delta_tanh_recurrent10_s256_b256_110k_noeval_20260227.log`

## 2026-02-27: GDH V2.3 plan+implementation (not launched yet)

### Versioning
- Designated next model variant as **GDH v2.3** (name correction from earlier "v4" suggestion).

### v2.3 changes implemented in code
1. **Init family parity with mainline**
   - Switched GDH read/write matrix init from `Normal(0,std)` to `Uniform(-s,s)` where `s = sqrt(3)*std` and `std = n_embd^-0.5`.
   - Applied to sidecar read/write matrices (including slot address tensors and write-brain input projection).

2. **ReZero outward sidecar coupling at init**
   - `W_o_read` now zero-initialized.
   - `W_write_mlp_out_global` now zero-initialized (when write-brain enabled).
   - **Kept `W_o_write` active** (uniform init) so internal sidecar state remains alive and gradients can flow.

3. **Read-mute floor adjustment**
   - Changed from `0.10 + 0.90*sigmoid(logit)` to:
   - `0.05 + 0.95*sigmoid(logit)` in forward path and telemetry probe.

4. **Optimizer policy retained**
   - GDH global/tied LR multiplier remains `1.0x` (full matrix LR), as in v2.2.

### Test updates + validation
- Updated GDH oracle/stability tests for v2.3 init contract and read-floor behavior.
- Validation command:
  - `PYTHONPATH=. .venv/bin/pytest -q tests/gdh/test_stability_controls.py tests/gdh/test_oracle.py`
- Result:
  - `60 passed`.

### Launch status
- Launched on request (no-eval, nohup detached).
- Run name: `gdh-v23-uniform-rezero-readfloor005-recurrent10-s256-b256-110k-noeval-20260227`
- W&B run: `mkohram-none/nanochat/mfzn2y1g`
- Log: `runlogs/gdh_v23_uniform_rezero_readfloor005_recurrent10_s256_b256_110k_noeval_20260227.log`
- Command:
  - `.venv/bin/python -m scripts.local_base_train --run gdh-v23-uniform-rezero-readfloor005-recurrent10-s256-b256-110k-noeval-20260227 --arch gdh --depth 4 --aspect-ratio 96 --head-dim 32 --max-seq-len 256 --device-batch-size 1 --total-batch-size 256 --num-iterations 110000 --dataset fineweb_edu --loader-mode recurrent_doc --recurrent-max-chunks-per-doc 10 --gdh-recurrent-carry-state --gdh-slots 20 --gdh-write-heads 12 --gdh-use-write-gate --gdh-write-gate-bias 1.0 --gdh-write-brain-hidden-mult 1 --no-value-embeds --eval-every -1 --core-metric-every -1 --sample-every -1 --save-every -1`

## 2026-02-27: GDH V2 + V2.1 stabilization pass

### V2 structural upgrades
- Slot-wise write gate (default): `Linear(D->R)` with scalar fallback (`D->1`).
- Read path switched to read-mute gate model:
  - `g_read_mute = sigmoid(x_read @ W_g_read_mute + b_g_read_mute)`
  - old read-competition gate path removed from active forward.
- EMA polarity fix:
  - in EMA mode gate output is retention `g`; do not pre-scale `delta` by `g`.

### V2.1 stabilization refinements
- Bias softening:
  - write retention bias run setting moved to `+1.0` (from `+3.0`)
  - read mute bias init moved to `-1.0` (from `-3.0`)
- Added pre-tanh governor:
  - `delta = tanh(RMSNorm(delta))` (controlled by `gdh_pre_tanh_rmsnorm` + `gdh_final_delta_tanh`).
- Telemetry naming clarity in EMA:
  - added `retention_gate_*` aliases for write-gate metrics.
- Added read-mute telemetry:
  - per-layer and aggregate `read_mute_mean/std/open_frac`.

### Test/audit outcomes
- Reworked GDH oracle tests to align with V2 contracts (removed legacy read-competition assumptions).
- Added missing regression tests for:
  - segmented leaky scan no-boundary stability,
  - write-gate shape contracts,
  - additive-vs-EMA gate polarity,
  - read-mute behavior.
- Current status:
  - `pytest -q tests/gdh` => `68 passed, 0 skipped, 0 failed`
  - full suite at patch time => `90 passed, 10 skipped` (FA3-dependent skips only).

## 2026-02-26: Stability controls (remedy2)

### Why
After remedy1, sidecar out-state magnitude still re-expanded later in training. We needed direct controls on update amplitude and high-leverage global dynamics.

### What changed
- Added **tanh write-value throttle** in GDH write paths:
  - `v_upd = tanh(x_write @ W_v_write)`
  - applied in both mainline helper (`nanochat/gpt.py`) and GDH core (`nanochat/double_helix.py`).
- Split GDH optimizer params into local vs global buckets in `GPT.setup_optimizer`:
  - global/tied tensors now use lower LR (`0.25 * matrix_lr`):
    - `W_k_read_global`, `W_v_read_global`, `E_slots`, `W_q_slots_global`, and write-brain globals when present.

### Tests added
- `tests/gdh/test_stability_controls.py`
  - `test_write_value_tanh_bounds_delta_range`
  - `test_ema_scan_with_tanh_delta_stays_bounded`
  - `test_gdh_global_params_use_lower_lr_group`

### Titanium Cage branch prep
- Created branch: `gdh-remedy3-titanium-cage`
- Added optional EMA scan mode (`--gdh-use-ema-scan`) wired through local train config.
- EMA scan uses write-gate retention (`g = sigmoid(gate)`), with vectorized segmented recurrence.
- For EMA mode, recurrent carry no longer applies extra 0.85 decay (to avoid double-decay).

---

## 2026-02-26: Recurrent-doc loader + BOS-aware segmented scan (GDH hygiene)

### Why
We found two issues for GDH experiments under BOS best-fit packing:
1. Long docs were often represented by only one prefix chunk in practice.
2. Sidecar accumulation needed strict BOS-boundary hygiene to prevent cross-doc bleed.

### What changed
- Added `nanochat/recurrent_dataloader.py` with contiguous per-doc chunking (`T+1` windows), capped by `max_chunks_per_doc`.
- Added `--loader-mode {bos_bestfit,recurrent_doc}` and `--recurrent-max-chunks-per-doc` in `scripts/local_base_train.py`.
- Added optional GDH chunk-to-chunk carry with detach (`--gdh-recurrent-carry-state`).
- Updated GDH scan in `nanochat/gpt.py` to BOS-aware segmented scan (vectorized; no per-token Python loop).

### Validation
- `PYTHONPATH=. pytest -q tests/gdh/test_oracle.py tests/test_recurrent_dataloader.py`
- Result: `60 passed, 6 skipped`

---

## Stage-2: The Masterpiece Run (Run 4.0)

**Result:** **PARTIAL SUCCESS** (22% Acc, 0.54 Cosine).
- Plateaued due to decay trap (beta < 1) + difficulty spike.

---

## Stage-3: The Victory Lap (Run 4.1)

**Objective:** Combine the working architecture with optimal physics to hit 100%.

**Code Patches (Active):**
1.  **Dead Slot Fix**: Usage loss on `alpha_soft`.
2.  **Garbage Penalty Fix**: Usage loss weighted by `g_write`.
3.  **Scope Fix**: Gate params in `_build_model`.

**Config:**
- `beta=1.0` (Perfect memory).
- `n_pairs=4` (Bridge difficulty).
- `steps=10,000` (Long runway).
- `vocab=128` (Disjoint).
- `swa=8` (Blindfolded).

### Run 4.1: Sparse GDH + Write Gate + Beta 1.0 + Patches
- **Status:** Running (PID 10406, session `rapid-cloud`)
- **Goal:** Acc -> 100%, Cosine -> ~0.60.
- **Results:**
    - Step 500: Acc 6.25%, Cos 0.83.
    - Step 2500: Acc 37.5%, Cos 0.88.
    - Final: [Pending]

---

## 2026-03-05: Minimal probe update + latest win (no write gate)

### Change
- Simplified `experiments/archive/scan-probe/mqar_scan_beta_probe_frozen_20260223.py` by removing probe write-gate path:
  - removed ad-hoc `model.g_write_projs` injection
  - removed `gate_proj` plumbing in `_write_delta_and_alpha`
  - removed `--disable-write-gate` flag (no longer needed)
- Probe GDH now runs **without** write gate by default in this frozen harness.

### Latest reproduced win (blind setup, no write gate)
- **Run file:** `experiments/out/mqar_scan_beta_probe_20260305_223059.json`
- **Config:**
  - `arch=gdh`, `beta=1.0`, `steps=5000`, `log_every=500`
  - `sequence_len=64`, `vocab_size=128`
  - `n_layer=3`, `n_head=4`, `n_embd=64`
  - `gdh_slots=8`, `route_topk=4`, `usage_balance_lambda=0.01`
  - `swa_window=8`, `n_pairs=2`, `n_queries=2`, `gap_min=16`, `gap_max=32`
  - vocab partition: `key=32`, `value=32`, `query_offset=64`, `filler_offset=96`, `filler_vocab=32`
- **Final metrics:**
  - `eval_acc_top1_last=0.6875`
  - `eval_acc_top5_last=0.9375`
  - `eval_mrr_last=0.8247`
  - `eval_ce_last=0.9933`
  - `slot_cos_last=0.8867`

### Baseline reference (same blind setup)
- `experiments/out/mqar_scan_beta_probe_20260305_220344.json`
- `eval_acc_top1_last=0.0`

**Takeaway:** In this easy blind regime, Top-K slot routing + sidecar path reproduces the win without requiring a write gate.
