# ACTIVE_GDH_PROBE_ARCHITECTURE

This document describes the current established GDH probe architecture in `experiments/`.

Scope:
- active, established probe behavior only
- no optional or inactive ablations
- no future-summary auxiliary loss
- no write cooloff
- no write brain
- no sparse top-k routing

The goal is to give an external auditor a clean description of what the probe is actually doing now.

## 1. What this probe is

This is a reduced GDH memory probe built on top of the current mainline `nanochat.gpt` transformer trunk.

It is not a production training path and not a full canonical GDH implementation. It is a narrow lab harness for studying:
- slot collapse
- write routing
- state accumulation
- long-gap recall under blindfolded MQAR
- behavior on real text via ClimbMix

Primary implementation:
- `experiments/mqar_gdh_mps_lab.py`

Primary active launchers:
- `experiments/run_probe_easy_mps.sh`
- `experiments/run_probe_hard_mps.sh`
- `experiments/run_probe_hard_climbmix_mps.sh`

## 2. Core architectural idea

The probe has one evolving GDH slot state per layer.

Per token position `t` and slot `r`, the layer maintains a state:
- `S[t, r] in R^D`

There is no separate persistent write-address state and no separate persistent value state in the active write path.

The active simplification is:
- layer 0 write routing is bootstrapped from learned slot seed vectors
- later layers route from the previous-layer sidecar state itself

So the architecture is moving toward a one-state story:
- the same slot state is what evolves over the sequence
- read routing and read values are both derived from that slot state
- write routing is seeded at layer 0, then becomes state-based in later layers

## 3. Transformer trunk

The probe reuses the current mainline GPT trunk.

High-level trunk properties inherited from `nanochat.gpt`:
- rotary position encoding
- QK norm
- `relu^2` MLP
- residual stream with learned `resid_lambdas` and `x0_lambdas`
- value embeddings enabled by default

The GDH logic is attached probe-locally and does not modify mainline GPT source behavior beyond adding probe-side modules and using the trunk activations.

## 4. Active defaults

### Easy MQAR launcher
From `experiments/run_probe_easy_mps.sh`:
- `sequence_len=64`
- `n_layer=3`
- `n_head=4`
- `n_embd=64`
- `gdh_slots=8`
- `gdh_write_heads=1`
- `route_topk=0`
- `write_routing=seeded`
- `state_mixer=normalized`
- `swa_window=8`
- `n_pairs=2`
- `n_queries=2`
- `gap_min=16`
- `gap_max=32`
- `batch_size=32`
- `eval_batch_size=8`
- `lr=3e-4`
- `device=mps`
- `compile=true`

### Hard MQAR launcher
From `experiments/run_probe_hard_mps.sh`:
- `sequence_len=256`
- `n_layer=4`
- `n_head=8`
- `n_embd=128`
- `gdh_slots=8`
- `gdh_write_heads=1`
- `route_topk=0`
- `write_routing=seeded`
- `state_mixer=normalized`
- `swa_window=8`
- `n_pairs=16`
- `n_queries=8`
- `gap_min=64`
- `gap_max=192`
- `batch_size=8`
- `eval_batch_size=8`
- `lr=3e-4`
- `device=mps`
- `compile=true`

### Hard ClimbMix launcher
From `experiments/run_probe_hard_climbmix_mps.sh`:
- inherits the hard launcher, then overrides:
- `data_source=climbmix`
- `gdh_slots=32`
- `gdh_write_heads=16`
- `swa_window=0`
- `eval_batch_size=2`
- `log_every=5`
- future-summary loss explicitly off

## 5. Per-layer computation

Let `x^l[t]` be the token stream entering layer `l`.
Let `S^l[t, r]` be the layer-`l` sidecar state at token `t` and slot `r`.

The active layer order is:

1. residual/input blend
- `x <- resid_lambda[l] * x + x0_lambda[l] * x0`

2. GDH read from current sidecar into the local stream
- `x <- Read_l(x, S)`

3. mainline transformer block
- `x <- Block_l(x)`

4. GDH write proposal
- `delta_l, alpha_l <- Write_l(x, S)`

5. tokenwise sidecar accumulation
- `S <- S + Mixer(delta_l, alpha_l)`

This repeats for every transformer layer.

After the final layer:
- `x <- gpt_norm(x)`
- `logits <- lm_head(x)`
- training loss is computed from logits

## 6. Read path

The active read path is content-based and single-head.

Given current token stream `x[t]` and current slot state `S[t, r]`:

1. token query
- `q_t = rms_norm(x_t) W_q_read`

2. slot keys and values from slot state
- `k_{t,r} = rms_norm(S_{t,r}) W_k_read`
- `v_{t,r} = rms_norm(S_{t,r}) W_v_read`

3. slot attention
- `alpha^read_{t,r} = softmax_r(q_t · k_{t,r} / sqrt(D))`

4. readout
- `z_t = sum_r alpha^read_{t,r} * v_{t,r}`
- `read_t = z_t W_o_read`

5. residual add
- `x_t <- x_t + read_t`

Important active behavior:
- `W_o_read` starts at zero, so the read branch is initially silent
- read-mute gate exists in code but is off by default in the active probe line

## 7. Write path

The active write path is multi-head capable, but the canonical easy/hard MQAR launchers currently use 1 write head.

### 7.1 Token-side write features
For token `t`:
- `k_t^w = rms_norm(x_t) W_k_write`
- `v_t^w = rms_norm(x_t) W_v_write`

If there are `H` GDH write heads, these are reshaped into head chunks:
- `k_{t,h}`
- `v_{t,h}`

### 7.2 Active seeded write routing
The active default routing mode is `seeded`.

Its meaning is:
- layer 0 routes from learned slot seed vectors
- later layers route from the previous-layer sidecar state only

#### Layer 0 routing state
For layer 0, the routing state is the learned slot seed table:
- `S_init[r] = E_slots[r]`

Routing logits are computed from these slot seed vectors.

#### Later-layer routing state
For layers `l > 0`, the routing state is the current previous-layer sidecar state:
- `routing_state[t, r] = S[t, r]`

So after layer 0, write routing is state-based rather than permanently tied to static slot anchors.

### 7.3 Routing equations
For routing state `R[t, r]` chosen as above:

1. slot routing vectors
- `a_{t,r} = rms_norm(R[t, r]) W_q_slots`

2. per-head routing logits
- `logits_{t,h,r} = a_{t,r,h} · k_{t,h} / sqrt(D_h)`

3. per-head slot routing weights
- `alpha_{t,h,r} = softmax_r(logits_{t,h,r})`

Because the active default is dense routing:
- no top-k masking is applied

### 7.4 Write content
Per-head routed write:
- `delta_head[t,h,r] = alpha_{t,h,r} * v_{t,h}`

Concatenate heads and project back to model width:
- `delta_raw[t,r] = concat_h(delta_head[t,h,r])`
- `delta[t,r] = delta_raw[t,r] W_o_write`

Important active behavior:
- `W_o_write` is alive at initialization
- write brain exists in code but is off by default in the active probe line

## 8. State accumulation: normalized mixer

The active probe line uses:
- `state_mixer=normalized`

This is the most important nontrivial part of the current architecture.

For each token `t` and slot `r`:

1. routed write content
- `delta[t, r]`

2. routed mass used for normalization
- `m[t, r] = mean_h alpha[t, h, r]`

Important:
- this is head-invariant
- each token contributes total routed mass `1` across slots, regardless of GDH write head count

3. tokenwise accumulation
With `beta=1.0` in the active launchers:
- `N[t, r] = sum_{tau <= t} delta[tau, r]`
- `D[t, r] = sum_{tau <= t} m[tau, r]`

4. normalized state contribution
- `C[t, r] = N[t, r] / max(D[t, r], eps)`

5. sidecar update
- `S[t, r] <- S[t, r] + C[t, r]`

Interpretation:
- this makes slot state behave like a routed running average, not a raw cumulative sum
- heavily written slots do not grow just because they were used many times
- the state tracks average written content per unit routed token mass

## 9. Why `normalized` was chosen

The active probe line converged to `normalized` because raw cumulative-sum state was too collapse-prone and too sensitive to repeated writes.

`normalized` changes the semantics of slot state from:
- total accumulated content

to:
- average content of the writes that selected that slot

That is the current established active choice.

## 10. Losses used in the active architecture

### MQAR
For MQAR, training uses next-token CE only at answer positions.

Sequence layout is:
- key/value pairs
- filler gap
- query/answer pairs

Targets are masked to `-100` except at answer positions.

So MQAR loss is:
- answer-only next-token cross-entropy

### ClimbMix
For ClimbMix, training uses ordinary next-token cross-entropy over BOS-packed text from the mainline ClimbMix data path.

### Not part of the active architecture
The following are intentionally not part of this architecture document:
- future-summary auxiliary loss
- write cooloff
- write brain
- sparse top-k routing
- read-mute gating

These may exist in code as optional ablations, but they are not part of the established active architecture described here.

## 11. Current architectural simplification, stated plainly

The key simplification compared with the earlier static-routing setup is:

Before:
- write routing depended on a permanent learned static slot-address system

Now:
- write routing is bootstrapped from learned slot seed state at layer 0
- after that, routing depends on the evolving sidecar state itself

This gives a cleaner and more symmetric story:
- read derives keys and values from the current slot state
- write, after bootstrap, also derives routing from the current slot state

So the architecture is moving toward:
- one evolving slot state `S`
- rather than a permanent static address object plus a separate memory object

## 12. Known limitations of the current established architecture

These are not optional features. These are real properties of the active design.

1. Layer-0 seeded routing may still be a weak symmetry breaker.
- All tokens see the same slot seed table at layer 0.
- Slots differ by seed vector, but there is no token-history-based slot differentiation yet.

2. Read is still single-head and purely content-derived from the current slot state.
- There is no separate read-address state.

3. Write and read both depend on the same evolving slot state after bootstrap.
- This is simpler, but it may still overload the state with both routing and value roles.

4. Collapse is still a live problem.
- The simplified architecture is cleaner, but current runs still show strong early slot-cosine collapse.

## 13. One-sentence summary

The active GDH probe is a mainline GPT trunk plus a per-layer slot memory in which write routing is seeded from learned slot seed vectors at layer 0, then becomes state-based, and slot state is updated by a head-invariant normalized routed running-average mixer rather than a raw cumulative sum.
