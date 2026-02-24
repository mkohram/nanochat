# GDH Slot-Collapse Audit (2026-02-21)

## Executive summary
We are **not** dealing with an `E_slots` collapse anymore.
The current failure mode is: **slot parameters stay distinct, but sidecar states still directionally re-merge**.

In plain terms: we made the slot addresses different, but the model still writes/accumulates memory in nearly the same direction across slots.

## What was audited

### 1) Structural identity fix
- Reintroduced global slot addresses (`E_slots`) + global slot query projection.
- Result: immediate hard collapse (`slot_std=0`, cosine=1) is gone.

### 2) `E_slots` prenorm in mainline
- Implemented `Q_slots = RMSNorm(E_slots) @ W_q_slots_global` in main code.
- Result: numerically fine, but no major change in long-run directional collapse.

### 3) `E_slots` Gram regularization (testbed sweep)
- Swept lambda from `1e-4` up to `1.0` (no noise, 1200 steps).
- Result: sidecar slot cosine stayed around `0.9995` / `0.9996`.
- Conclusion: did not solve collapse.

### 4) Memory GLU + `V_slots` Gram regularization (testbed)
- Added testbed-only Memory GLU path:
  - `V_slots = RMSNorm(E_slots) @ W_v_static_global`
  - `V_filter = 1 + gamma * tanh(V_slots)`
  - slot-conditioned filtered values used for write updates.
- Added `V_slots` Gram loss (`--vslots-gram-lambda`) and swept `0.0 -> 1.0`.
- Result: still sidecar cosine around `0.9995` / `0.9997`.
- Conclusion: also did not solve collapse by itself.

### 5) Matrix-level distinctness checks
- Checked pairwise cosine for:
  - `E_slots` rows
  - `Q_slots = RMSNorm(E_slots) @ W_q_slots_global`
  - `V_slots`
- Result:
  - `E_slots` and `Q_slots` are distinct even with no Gram loss.
  - With `V_slots` Gram loss, `V_slots` becomes near-orthogonal as intended.
  - But sidecar states still re-merge directionally.

### 6) Gamma sensitivity check (Memory GLU)
- Forcing bigger fixed gamma (0.5, 1, 2, 5, 10) reduces collapse a bit, but does not remove it.
- Tradeoff: CE worsens slightly at high gamma.

## Core finding
We are mostly patching **parameter geometry**, while the collapse is in **state dynamics**.

The model can keep slot parameters distinct and still write redundant information into sidecar states.

## Why this can happen
1. CE loss does not strongly reward slot-level specialization.
2. Shared write/output transforms plus additive accumulation can still align slot directions.
3. Per-token updates remain strongly coupled by shared token value dynamics.
4. Parameter-level orthogonality is not equivalent to state-level orthogonality.

## Positive control already seen
- In prior isolated probes, explicit state-level orthogonality pressure can keep slots distinct.
- This suggests the issue is not impossibility; it is objective placement.

## Most likely next correct direction
Target **state-level** losses, not only parameter-level losses:
- Gram/off-diagonal penalty on sidecar state `S` (sampled timesteps/layers), and/or
- Gram on write updates `delta` after write mixer, and
- keep routing/usage diagnostics to verify specialization actually appears.

## Litmus test on "+1 baseline" initialization hypothesis
A controlled override was run on the same trained Memory GLU model:
- default: `V_filter = 1 + gamma*tanh(V_slots)`
- tanh-only (no `+1`, forced gamma=1): `V_filter = tanh(V_slots)`
- plus+gamma1: `V_filter = 1 + tanh(V_slots)`
- centered `V_upd` with default filter

Observed on trained model:
- default sidecar cosine stayed very high (~0.99964 / 0.99977 by layer)
- tanh-only reduced cosine slightly (~0.99905 / 0.99958) but still strongly collapsed
- centered `V_upd` had negligible effect

Interpretation:
- The "+1 baseline" affects early behavior, but it is not the sole root cause of long-run directional collapse.

## Sparse-routing sanity test (top-k mask)
A quick testbed-only sparse routing variant was added (post-softmax top-k mask + renorm):
- `route_topk=0` (dense): sidecar cosine ~`0.99952 / 0.99970`
- `route_topk=2`: sidecar cosine ~`0.99770 / 0.99851`
- `route_topk=1`: sidecar cosine ~`0.98552 / 0.99057`

Interpretation:
- Sparse routing *does* reduce collapse in proportion to sparsity.
- But hard top-1 introduces near-zero gradients on some routing matrices in this implementation, so it is likely too aggressive without a smoother sparse mechanism.

## Scale sanity check (is the sandbox too small?)
A one-off larger-sandbox run was compared against the default small probe under pure cumsum:
- small: `D=64, layers=2, seq=64`, 300 steps -> last-layer slot cosine `0.361 -> 0.99918`
- larger: `D=128, layers=4, seq=256`, 300 steps -> last-layer slot cosine `0.552 -> 0.99976`

Interpretation:
- Collapse persists in a larger sandbox and did not weaken in this test.
- So "toy size only" is unlikely to be the primary explanation.

## Leaky-scan diagnosis (testbed)
On the cleaned testbed (no Memory GLU), a leaky accumulator was added:
- `scan_beta=1.0`: pure cumsum
- `scan_beta<1.0`: `y_t = beta*y_{t-1} + delta_t`

1200-step no-noise runs showed monotonic collapse reduction as beta decreased:
- beta 1.0: sidecar cosine ~`0.99949 / 0.99964`
- beta 0.95: sidecar cosine ~`0.99907 / 0.99946`
- beta 0.90: sidecar cosine ~`0.99873 / 0.99924`
- beta 0.50: sidecar cosine ~`0.98847 / 0.99239`

Interpretation:
- The infinite-integrator effect is real and measurable.
- Mild leakage helps only slightly; stronger leakage yields clear separation gains.

## MQAR (forced-memory) initial scan-beta sweep
A synthetic MQAR harness was implemented with:
- masked CE only on answer positions (`ignore_index=-100`)
- KV pairs + variable filler gap + queries in the same sequence
- capacity stress (`n_pairs > gdh_slots`)

Initial 300-step sweep (betas `1.0, 0.99, 0.95, 0.9`) produced:
- eval MQAR accuracy: `0.0` for all betas
- masked CE: ~`7.74` for all betas
- last-layer slot cosine: `1.0` for all betas by step 300

Follow-up stronger control (600 steps; betas `1.0, 0.7, 0.5`) produced:
- eval MQAR accuracy: still `0.0` for all betas
- masked CE: ~`7.689` to `7.692`
- last-layer slot cosine: still `1.0` at step 600

Interpretation:
- In this regime, decay in `0.5..1.0` did not recover MQAR performance and did not prevent collapse.
- This points to needing additional mechanisms (state-level anti-collapse objective, routing dynamics changes, or task/harness refinements) beyond scalar leak alone.

Additional control:
- A vanilla `arch=baseline` transformer under the original hard MQAR setup and 600-step budget also stayed at top-1 accuracy `0.0` while CE dropped similarly.
- This confirmed the initial harness/budget was too hard to be discriminative.

Stage-0 curriculum validation (new):
- Built an easier Stage-0 MQAR config (`seq=32, vocab=128, pairs=2, queries=2, gap=0`, 3000 steps).
- Baseline now learns strongly (top-1 ~`0.992`, MRR ~`0.996`), so assay is valid.
- GDH results on Stage-0:
  - dense routing (`topk=0`) underperforms (top-1 ~`0.789`, slot-cos ~`0.91`)
  - sparse routing (`topk=2`/`4`) recovers top-1 ~`0.99`
  - adding usage-balance loss further reduces slot cosine (down to ~`0.62` at `topk=4 + balance`), with small top-1 cost (~`0.984`)

Interpretation update:
- Sparse top-k routing is strongly supported in this validated Stage-0 setting.
- Usage balancing improves slot separation further.
- Dense routing remains the weakest GDH condition under this curriculum.

## Recommendation
Treat `E_slots`/`V_slots` regularization as secondary conditioning priors.
For the main collapse fix, move the objective to the tensors that actually collapse (`S`, `delta`) and consider leaky accumulation and/or smoother sparse-routing mechanisms.
