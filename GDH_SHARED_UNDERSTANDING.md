# GDH Shared Understanding (Implementation Notes)

Date: 2026-02-18

This file captures the *intent* behind GDH so implementation stays aligned even if formal math is revised.

## Core intent

- Architecture flow: **Read -> Process -> Write**.
- Local stream `L` remains standard transformer token representations.
- Sidecar `S` is a latent global memory updated across sequence positions.

## Read intent (Global -> Local)

- Each token creates a local query via a layer-specific translator (base + LoRA dialect).
- Sidecar provides context via global read K/V projections.
- Retrieved context is gated and injected into local token state.

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

- Keep write path compatible with parallel training (prefix/cumsum style accumulation).
- No-forget-gate in v1 (additive state); handle stability via normalization + reset policy.

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

- Added a second slow oracle (`tests/gdh/oracle_decomposed.py`) with **shared context-side decomposition**.
- One shared long/context factor `U_ctx_shared` is used across Q_read / K_write / V_write translator paths.
- Translator-specific short/context factors (`V_ctx_*`) define per-path behavior.
- Causal sequence mixers are built as masked row-softmax of `U_ctx_shared @ V_ctx_*`.
- This keeps the implementation explicit and testable while reflecting the "share long (context) side" design intent.
