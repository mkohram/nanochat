# MQAR Experiment Log

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
