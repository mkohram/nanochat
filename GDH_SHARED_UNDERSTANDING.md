# GDH Shared Understanding (Implementation Notes)

Date: 2026-02-18 (updated 2026-02-27)

This file captures the *intent* behind GDH so implementation stays aligned even if formal math is revised.

## Core intent

- Architecture flow: **Read -> Process -> Write**.
- Local stream `L` remains standard transformer token representations.
- Sidecar `S` is a latent global memory updated across sequence positions.

## Read intent (Global -> Local)

- Each token creates a local query via a layer-specific translator (base + LoRA dialect).
- Sidecar provides context via global read K/V projections.
- Retrieved context is injected through a simple read-mute gate (`sigmoid`) before residual add.

## Process intent (Local refinement)

- Standard causal transformer processing (self-attn + MLP path).
- Token `t` output is **prefix-conditioned** (depends only on tokens `1..t`).

## Write intent (Local -> Global)

- Processed local token produces write K/V via layer-specific translator (base + LoRA dialect).
- Global write Q (slot addressing template) routes update into sidecar slots.
- Per-token write proposal is accumulated with prefix scan over sequence.

## Causality / worldview requirement

- At training and inference, token `t` should only use information available up to `t`.
- The model should learn a "world view up to now," never future leakage.

## Parallel training intent

- Keep write path compatible with parallel training (vectorized segmented-prefix accumulation).
- Use canonical gated EMA scan (slot-wise retention) while preserving sequence-parallel execution.

## Reset policy intent (v1)

- Sidecar must reset at document/chunk boundaries.
- If no boundary metadata exists yet, treat each training chunk as its own segment.

## Scope guardrails

- Keep base scripts/modules clean where possible.
- New core logic lives in `nanochat/double_helix.py`.
- `base + LoRA` meaning: **base is shared across all tokens within a layer**.

## Clarifications from review (must preserve)

- There is **one local state per token** per layer (`L` shape conceptually `(B, N, D)`), not `N` copies per token.
- There is **one write contribution per token** (`Δ_t`), and sidecar state at `t` is the prefix accumulation over prior contributions.
- In current v1.4 training math, `S_{t,0}=0` is the layer-0 anchor (parallel layer scan formulation).
- There is **one read output per token** (`Z_t`), aligned to that token's sidecar snapshot.
- Writes are **parallel and prefix-conditioned** (through causal Process phase), not token-independent in the causal sense.
- Current v1.4 routing is static on the slot-query side (`Q_slots` from slot embeddings), but still indirectly context-aware through token-conditioned `K_upd/V_upd`.
- Training objective intent: token `t` should form/update a world-view using only information available up to `t` (no future leakage).

## Decomposed-oracle variant (added)

- Added a second slow oracle path in `tests/gdh/oracle.py` with **shared context-side decomposition**.
- One shared long/context factor `U_ctx_shared` is used across Q_read / K_write / V_write translator paths.
- Translator-specific short/context factors (`V_ctx_*`) define per-path behavior.
- Causal sequence mixers are built as masked row-softmax of `U_ctx_shared @ V_ctx_*`.
- This keeps the implementation explicit and testable while reflecting the "share long (context) side" design intent.

## Current mainline implementation snapshot (2026-02-23)

- Write routing in `nanochat/double_helix.py` uses **learnable slot addresses**:
  - `Q_slots = RMSNorm(E_slots) @ W_q_slots_global` (global/shared)
  - `K_upd = x_write @ W_k_write`, `V_upd = x_write @ W_v_write` (layer-local)
- Read phase still attends to accumulated sidecar state (`S`) via global read K/V.
- Mainline temporal integration in `nanochat/gpt.py` is canonical EMA scan:
  - `S_t = g_t * S_{t-1} + (1-g_t) * delta_t` (slot-wise retention gate)
- EMA scan implementation is associative + vectorized (parallel tensor ops), boundary-aware, and uses implicit token-0 segment start when no boundary appears yet.
- Write path applies a single final bound point before accumulation:
  - `delta = tanh(RMSNorm(delta))`
  - no additional tanh in earlier `v_upd` path.
- Write gate is slot-wise (`Linear(D->R)`) in canonical path.
- Gate output is retention (`g`), so `delta` is not pre-multiplied by `g`.
- Read-competition gate path is removed from active forward; read injection uses read-mute gate (`W_g_read_mute`, `b_g_read_mute`).
- Optimizer uses a lower LR bucket for tied/global GDH tensors (`W_k_read_global`, `W_v_read_global`, `E_slots`, `W_q_slots_global`, and write-brain globals when enabled).
- Layer-sharing policy in `nanochat/gpt.py`:
  - shared across layers: `W_k_read_global`, `W_v_read_global`, `E_slots`, `W_q_slots_global`
  - layer-local: `W_k_write`, `W_v_write`, `W_o_write`, read-local params
- Canonical naming cleanup:
  - use `W_q_slots_global` (not `W_q_write_global`) for the global slot-query projection.
  - `GDHWriteCore` keeps a legacy alias/loading shim so older checkpoints still restore.
- Telemetry in `scripts/local_base_train.py` logs global write-routing diagnostics for `E_slots` and `W_q_slots_global` (including tie checks across layers).
- Telemetry uses retention-gate naming (`retention_gate_*`) in canonical EMA path to avoid polarity confusion.
- Read-mute telemetry is available per-layer and aggregate (`read_mute_*`).

## MQAR probe-only extensions (current testbed)

These were active in the archived Stage-1 blindfold probe `experiments/archive/scan-probe/mqar_scan_beta_probe.py`, but are **not** part of the canonical GDH spec yet.

- **Sparse routing (`route_topk`)**
  - Applied post-softmax on slot routing weights, then renormalized.
  - Used to reduce effective write density and slot collisions.

- **Usage balancing loss (`usage_balance_lambda`)**
  - Layer-local auxiliary loss encouraging uniform slot usage.
  - Implemented on `alpha_soft` (pre-topk hard mask), with gate-aware weighting.

- **Ad-hoc write gate injection (`g_write`) in `_build_model`**
  - Probe injects per-layer scalar gate modules (`Linear(D->1)`) as `model.g_write_projs`.
  - Bias initialized negative (e.g. `-2.0`) so writes start mostly closed.
  - Final write delta is multiplied by `sigmoid(g_write)`.
  - Same gate weighting is used in usage-balancing aggregation so low-gate/noisy tokens contribute less to slot-usage pressure.

## Recurrent-doc loader mode (implemented for local training)

- `scripts/local_base_train.py` now supports `--loader-mode recurrent_doc`.
- In recurrent mode, each document is split into contiguous `T+1` chunks and emitted in-order.
- `--recurrent-max-chunks-per-doc` caps how many chunks are taken from one document.
- Optional GDH chunk-to-chunk state carry is available via `--gdh-recurrent-carry-state`.
- Carry state is detached each step (TBPTT style) and BOS-aware.
- Canonical EMA path keeps carry unscaled at handoff (detached, BOS-aware).

## Future ideas backlog (not implemented yet)

- **Sidecar transition auxiliary loss** (self-supervised sidecar objective):
  - Let current sidecar state predict the **next write delta** (`Δ_{t+1}`), instead of only relying on token CE loss.
  - Preference: predict next-**delta** rather than next-state (`S_{t+1}`), because next-state can become an identity shortcut (`S_{t+1}=S_t+Δ_{t+1}`).
  - Candidate form:
    - predictor input: `RMSNorm(S_t)`
    - predictor output: `\hat{Δ}_{t+1}`
    - target: `stopgrad(RMSNorm(Δ_{t+1}))`
    - loss: cosine/Huber
  - Integrate as small weighted auxiliary term: `L = L_ce + λ * L_sidecar` (warm up `λ` conservatively).
  - Start with a **global/shared predictor** in sidecar space for symmetry with shared sidecar transforms.
