# GDH (Gated Double Helix) — Current Architecture Spec

Last updated: 2026-02-23

This spec reflects the behavior currently implemented in:
- `nanochat/gpt.py` (mainline GDH path)
- `nanochat/double_helix.py` (GDH read/write cores)
- `experiments/mqar_scan_beta_probe.py` (probe/testbed extensions)

---

## 1) Symbols and core knobs

- `B`: batch size
- `T`: sequence length
- `D`: model width
- `L`: number of transformer/GDH layers
- `R`: number of sidecar slots
- `h`: GDH write heads
- `d_h = D / h`

Mainline GDH knobs (`GPTConfig`):
- `gdh_slots = R`
- `gdh_write_heads = h` (or fallback to `n_head`)
- `gdh_use_read_gate` (bool)
- `gdh_use_write_brain` (bool)
- `gdh_write_brain_hidden_mult` (int)

Probe-only knobs (`mqar_scan_beta_probe.py`):
- `route_topk = K`
- `usage_balance_lambda = λ_usage`
- `scan_beta = β`
- ad-hoc per-layer `g_write` projections

---

## 2) Parameter taxonomy (what is global vs local)

### 2.1 Shared across layers (tied globals)

From `GPT._tie_gdh_global_weights()`:
- Read globals:
  - `W_k_read_global ∈ R^{D×D}`
  - `W_v_read_global ∈ R^{D×D}`
- Write globals:
  - `E_slots ∈ R^{R×D}`
  - `W_q_slots_global ∈ R^{D×D}`
- If write-brain is enabled:
  - `W_write_mlp_in_global ∈ R^{D×(mD)}`
  - `W_write_mlp_out_global ∈ R^{(mD)×D}`
  where `m = gdh_write_brain_hidden_mult`.

### 2.2 Layer-local GDH params

Per layer `i`:
- Read local:
  - `W_q_read`, `W_o_read`, `W_g_read`, `W_g_side`
  - read-gate scalar terms used in current gate logic:
    - `w_g_interaction`, `w_g_confidence`, `w_g_novelty`,
    - `w_g_synergy`, `w_g_querymatch`, `w_g_queryadv`, `w_g_queryadv2`,
    - `w_g_temp`, `w_g_temp_adv`
- Write local:
  - `W_k_write`, `W_v_write`, `W_o_write`

(Additional deprecated `w_g_*` scalars exist for compatibility but are not active in current read-gate math.)

---

## 3) Per-layer forward order (mainline)

For each layer `i`, with local stream `X` and sidecar stream `S`:

1. **Residual blend**
   - `X ← resid_lambda_i * X + x0_lambda_i * X0`

2. **Read (sidecar → local)**
3. **Process (standard transformer block)**
4. **Write proposal (local → sidecar delta)**
5. **Temporal accumulation**
   - `S ← S + cumsum(Δ, dim=time)`

Important: current mainline order is **Read → Process → Write** (not Process → Write → Read).

---

## 4) Read phase (current implementation)

Inputs:
- `X ∈ R^{B×T×D}`
- `S_prev ∈ R^{B×T×R×D}`

Core retrieval:
- `x_read = RMSNorm(X)`
- `q = x_read W_q_read`                             (`B×T×D`)
- `s_hat = RMSNorm(S_prev)`
- `k = s_hat W_k_read_global`                       (`B×T×R×D`)
- `v = s_hat W_v_read_global`                       (`B×T×R×D`)
- `α_read = softmax((q·k)/sqrt(D), dim=R)`          (`B×T×R`)
- `z = Σ_r α_read[r] v[r]`                          (`B×T×D`)

Output fusion:
- If `gdh_use_read_gate = False`:
  - `X_read = X + z W_o_read`
- Else (2-way competition gate):
  - local logit from `W_g_read`
  - side logit from `W_g_side` + active scalar terms listed above
  - adaptive temperature from confidence/query-advantage terms
  - `g_read = softmax([local_logit, side_logit] / temp)[side]`
  - `X_read = X + g_read * (z W_o_read)`

---

## 5) Process phase

`X_proc = TransformerBlock(X_read)`

(Attention + MLP are already inside this block in `gpt.py`.)

---

## 6) Write phase (mainline core)

From `X_proc`:
- `x_write = RMSNorm(X_proc)`
- `k_upd = x_write W_k_write`                       (`B×T×D`)
- `v_upd = x_write W_v_write`                       (`B×T×D`)

Slot queries:
- `q_slots = RMSNorm(E_slots) W_q_slots_global`     (`R×D`)

Multi-head routing (per head width `d_h`):
- `logits[b,t,head,r] = <q_slots[r,head], k_upd[b,t,head]> / sqrt(d_h)`
- `α = softmax(logits, dim=r)`
- `Δ_raw` built by weighted `v_upd` across slots
- `Δ = Δ_raw W_o_write`                              (`B×T×R×D`)

Optional write-brain (if enabled):
- `Δ ← Δ + MLP(RMSNorm(Δ))`, with `Linear → ReLU² → Linear`

Mainline temporal update:
- `S ← S_prev + cumsum(Δ, dim=1)`

---

## 7) Initialization facts (as implemented)

- `E_slots ~ N(0, 1)`
- Read mixer `W_o_read` is zero-init at model init (Option-1 bootstrap)
- Write mixer `W_o_write` is non-zero init (allows early sidecar writes)
- If write-brain enabled:
  - `W_write_mlp_in_global` normal init
  - `W_write_mlp_out_global` zero-init (safe residual bootstrap)

---

## 8) MQAR probe extensions (testbed-only, not canonical mainline)

In `experiments/mqar_scan_beta_probe.py`:

1. **Probe GDH config forcing + read warm-start**
   - In `_build_model`, probe currently hard-sets `gdh_use_write_brain=True` and `gdh_write_brain_hidden_mult=4`.
   - `gdh_use_read_gate` is passed through from CLI (`--gdh-use-read-gate`).
   - After `model.init_weights()`, the probe re-initializes `W_o_read` for the first `min(2, n_layer)` GDH read modules with `Normal(0, 0.02)`.

2. **Sparse routing (`route_topk`)**
   - Start with `alpha_soft = softmax(logits)`.
   - If `0 < K < R`, mask to top-`K` slots and **renormalize** masked weights so selected slots sum to 1.
   - Otherwise (`K<=0` or `K>=R`), use dense routing (`alpha = alpha_soft`).

3. **Ad-hoc write gate (`g_write`)**
   - Injects per-layer `Linear(D→1)` modules in `_build_model`.
   - Bias initialized to `-2.0`.
   - `Δ ← Δ * sigmoid(g_write)`.

4. **Gate-aware usage balancing loss (`usage_balance_lambda`)**
   - Uses `alpha_soft` (not hard-masked `alpha`) for gradient flow to low-usage slots.
   - Uses detached gate weights when present:
     - `gate_weight = g_write.detach()` (else ones)
     - `weighted_sum_r = Σ_{b,t,h} alpha_soft[b,t,h,r] * gate_weight[b,t,1,1]`
     - `usage_r = weighted_sum_r / (gate_weight.sum() * h + 1e-9)`
   - Layer loss:
     - `L_usage_layer = mean_r (usage_r - 1/R)^2`
   - Total:
     - `L_total = CE + λ_usage * mean_layers(L_usage_layer)`

5. **Leaky scan option (`scan_beta`)**
   - `β = 1.0`: pure `cumsum`.
   - `0 < β < 1`: leaky recurrence `y_t = β y_{t-1} + Δ_t` (vectorized closed form).

6. **Blindfold harness**
   - `WindowedGPT` forces SWA window on **all** layers (including final layer).

---

## 9) Explicit non-claims (to avoid drift)

The following are **not** current guaranteed mainline behavior:
- Mandatory Shazeer `P_mean · f_mean` load-balance formula
- Mandatory no-renorm top-k masking rule
- Mandatory global-parameter LR scaling by `1/sqrt(L)`
- Mandatory read-gate bias `-2.0` in core model params

Those may be explored, but are not the current code contract.
