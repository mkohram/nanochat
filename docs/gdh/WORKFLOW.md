# GDH Workflow (Local + Completion)

Last updated: 2026-02-23

This file combines:
- local project memory for this machine/repo
- GDH completion checklist before marking work done

## 1) Local runtime memory (GTX 1060 3GB safe defaults)

For stable `scripts/local_base_train.py` runs on this machine:

- `--device-batch-size=1`
- `--total-batch-size=256`
- `--num-iterations=10000` (or as needed)
- `--core-metric-every=-1`
- `--eval-every=-1`
- `--sample-every=-1`

Why:
- Larger micro-batches (e.g. `--device-batch-size=32`) OOM on GTX 1060 3GB.
- Typical idle VRAM (~250MB used) indicates OOMs are generally config-related.

Canonical safe launch:

```bash
source .venv/bin/activate
python -m scripts.local_base_train \
  --run baseline-d10-s256-10k-safe-<timestamp> \
  --num-iterations=10000 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --core-metric-every=-1 \
  --eval-every=-1 \
  --sample-every=-1
```

Notes:
- Keep `scripts/base_train.py` and `scripts/base_eval.py` clean for upstream/scale behavior.
- Use `scripts/local_base_train.py` and `scripts/local_base_eval.py` for local GTX 1060 work.

## 2) Architecture guardrails

- Do not break existing baseline path.
- Keep GDH logic in `nanochat/double_helix.py`.
- Wire GDH behind optional config/flags.
- Keep docs aligned with implementation (`SPEC.md`, `GDH_SHARED_UNDERSTANDING.md`).

## 3) Completion checklist (required before saying "done")

### A. Correctness tests
Run:

```bash
.venv/bin/python -m pytest -q tests/gdh
```

Pass condition:
- tests pass (skips allowed if explicitly intentional and documented)

Shortcut:

```bash
scripts/gdh_checklist.sh
```

### B. Compatibility checks
- If naming/state keys changed, ensure legacy checkpoint loading still works.
- Keep migration/alias coverage in the GDH suite (e.g., `tests/gdh/test_oracle.py`).

### C. Docs sync
When equations/behavior changed, update:
- `SPEC.md`
- `GDH_SHARED_UNDERSTANDING.md`

### D. Memory/logging
Append a short note to workspace memory (`memory/YYYY-MM-DD.md`) with:
- what changed
- test commands run
- pass/fail summary

### E. Chat report format
When reporting completion, include:
1. code files changed
2. test command(s) run
3. exact pass summary
4. docs updated

## 4) Current mainline GDH note

- Write routing uses learnable slot addresses (`E_slots`) + global slot query projection (`W_q_slots_global`).
- Layer-local write translators remain `W_k_write` / `W_v_write` / `W_o_write`.
- This was adopted to address slot symmetry collapse while preserving sequence-parallel write accumulation.
