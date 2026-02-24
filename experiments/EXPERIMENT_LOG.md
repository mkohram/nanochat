# MQAR Experiment Log

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
