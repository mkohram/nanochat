# GDH Step-by-Step Plan (Test-First, Slow Execution)

Date: 2026-02-18
Status: In progress (oracle + scaffold complete; slot-address write routing now implemented in mainline)

## Working mode
- Pace: deliberate, step-by-step
- Principle: **no coding before tests + contract checks**
- Scope: v1.4 spec, **no forget gate**

## Guardrails
1. Keep baseline modules/scripts clean.
2. New GDH logic lives in `nanochat/double_helix.py`.
3. Preserve causality: token `t` only uses prefix information.
4. Keep implementation aligned with shared notes (`GDH_SHARED_UNDERSTANDING.md`).

## Progress snapshot
- ✅ Dense + decomposed slow oracles implemented in a single file (`tests/gdh/oracle.py`) and tested
- ✅ Cross-oracle equivalence tests added
- ✅ `nanochat/double_helix.py` scaffold added and upgraded to dense v1 core
- ✅ Tests consolidated into one file (`tests/gdh/test_oracle.py`)
- ✅ Minimal GPT integration behind `GPTConfig.arch` (`baseline`|`gdh`), default remains baseline

---

## Phase 0 — Contract freeze (first)
Create an explicit checklist from spec that all later code/tests must satisfy:
- one write proposal per token (`Δ_t`)
- one read output per token (`Z_t`)
- writes are prefix-conditioned via causal Process phase
- sidecar accumulation uses prefix-scan/cumsum semantics
- no forget gate in v1
- boundary reset policy exists (chunk-level for now)

**Exit criterion:** checklist reviewed and approved.

---

## Phase 1 — Reference oracle (slow, unoptimized)
Create a tiny reference implementation inside tests (loop-based over time):
- clear read/process/write order
- explicit causal behavior
- explicit sidecar update per token

This oracle is correctness truth for all optimized code.

**Exit criterion:** oracle tests pass on CPU for tiny shapes.

---

## Phase 2 — Unit test suite (rigorous)
Add GDH tests before production implementation:
1. `test_gdh_shapes.py`
2. `test_gdh_causality.py`
3. `test_gdh_prefix_accumulation.py`
4. `test_gdh_reset_boundaries.py`
5. `test_gdh_zero_init_noop.py`
6. `test_gdh_parallel_equals_reference.py`
7. `test_gdh_backward_stability.py`

**Exit criterion:** tests are in place (initially failing allowed), then moved to green incrementally.

---

## Phase 3 — Module scaffolding only
Create `nanochat/double_helix.py` with interfaces and docstrings only:
- config dataclass/typed settings for GDH
- function/class stubs for read/process/write helpers
- clear tensor-shape comments

No integration into training path yet.

**Exit criterion:** imports compile, no behavior changes.

---

## Phase 4 — Implement minimal GDH core behind tests
Implement in smallest vertical slices:
1. write proposal path
2. prefix accumulation
3. read path + gated injection
4. combined block forward for tiny tensors

Validate each slice against oracle.

**Exit criterion:** optimized path matches oracle within tolerance.

---

## Phase 5 — Controlled integration
After core correctness only:
- wire optional GDH path behind explicit flag
- keep baseline path untouched
- run tiny train smoke (10-20 steps)

**Exit criterion:** baseline still works; GDH path runs without NaN/shape break.

---

## Non-goals for now
- performance tuning
- full-scale training
- architecture variants (forget gate, dynamic slot query, etc.)

These come after correctness.
