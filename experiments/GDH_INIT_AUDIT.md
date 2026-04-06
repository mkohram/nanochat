# GDH Init Audit vs Mainline Transformer Init

This note audits the **current GDH-side parameter initializations** in `experiments/mqar_gdh_mps_lab.py` against the **mainline transformer init patterns** in `nanochat/gpt.py`.

Goal:
- identify which GDH params already follow mainline init conventions
- identify which ones differ
- separate "strict mainline analogy" from "probably intentional deviation for GDH viability"

## Source of truth

- GDH init:
  - `experiments/mqar_gdh_mps_lab.py`
  - `LabGDHReadCore.reset_parameters(...)`
  - `LabGDHWriteCore.reset_parameters(...)`
  - `_build_model(...)`
- Mainline transformer init:
  - `nanochat/gpt.py`
  - `GPT.init_weights()`

## Mainline init patterns to compare against

Let:

- `std_main = n_embd^-0.5`
- `s = sqrt(3) * std_main`

Current mainline transformer uses:

- attention q/k/v:
  - `Uniform(-s, s)`
- attention output projection (`c_proj`):
  - zeros
- MLP input (`c_fc`):
  - `Uniform(-0.4s, 0.4s)`
- MLP output (`c_proj`):
  - zeros
- value embeddings:
  - `Uniform(-s, s)`
- VE gate weights:
  - `Uniform(0.0, 0.02)`
- most mainline linear layers are bias-free

## Current GDH init summary

The GDH lab uses:

- `gdh_std = n_embd^-0.5`
- internally `s = sqrt(3) * gdh_std`

So wherever GDH uses `Uniform(-s, s)`, it is matching the mainline q/k/v/value-embed scale.

## Read core audit

### `W_q_read` `[D, D]`
Role:
- local stream -> read query

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_q`

Verdict:
- aligned with mainline

### `W_k_read_global` `[D, D]`
Role:
- sidecar state -> read key

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_k`

Verdict:
- aligned with mainline

### `W_v_read_global` `[D, D]`
Role:
- sidecar state -> read value

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_v`

Verdict:
- aligned with mainline

### `W_o_read` `[D, D]`
Role:
- read output -> residual stream

Current GDH init:
- zeros

Closest mainline analog:
- attention `c_proj`
- MLP output projection

Verdict:
- aligned with mainline

### `W_g_read_mute` `[D, 1]`
Role:
- tokenwise scalar mute gate for read injection

Current GDH init:
- zeros

Closest mainline analog:
- no exact analog
- nearest explicit gate-like analog in mainline is `ve_gate.weight`, which uses `Uniform(0.0, 0.02)`

Verdict:
- not obviously mainline-like
- current zero init is more conservative than the VE gate pattern

### `b_g_read_mute` `[1]`
Role:
- bias for read mute gate

Current GDH init:
- `-1.0`

Closest mainline analog:
- none; mainline gate layers are generally bias-free

Verdict:
- explicit GDH-specific deviation

## Write core audit

### `E_slots` `[R, D]`
Role:
- learned slot addresses / memory anchors

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- value embeddings, or more loosely other learned state-space embeddings

Verdict:
- reasonably mainline-like
- this is the init I would keep if the goal is mainline consistency

### `W_q_slots_global` `[D, D]`
Role:
- slot address -> slot query

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_q` / `c_k`

Verdict:
- aligned with mainline

### `W_k_write` `[D, D]`
Role:
- local stream -> write key

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_k`

Verdict:
- aligned with mainline

### `W_v_write` `[D, D]`
Role:
- local stream -> write value

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention `c_v`

Verdict:
- aligned with mainline

### `W_o_write` `[D, D]`
Role:
- routed write payload -> sidecar update basis

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- attention output projection `c_proj`
- MLP output projection

Strict mainline pattern would be:
- zeros

Verdict:
- this is a real mismatch versus mainline init style
- however it is also likely an intentional one

Important note:
- if `W_o_write` were zero **and** `W_o_read` is zero, the GDH path starts doubly silent:
  - no write reaches sidecar
  - no read reaches residual stream
- with zero-initialized sidecar, that creates a serious risk of a dead GDH branch at step 0

So:
- **strict mainline analogy says zero**
- **GDH viability likely prefers nonzero here**

### `W_write_mlp_in_global` `[D, hidden]`
Role:
- write-brain input projection

Current GDH init:
- `Uniform(-s, s)`

Closest mainline analog:
- MLP input `c_fc`

Strict mainline pattern would be:
- `Uniform(-0.4s, 0.4s)`

Verdict:
- mismatch versus mainline
- if we want write-brain init to track trunk MLP style, this should probably change to `0.4x` scale

### `W_write_mlp_out_global` `[hidden, D]`
Role:
- write-brain output projection

Current GDH init:
- zeros

Closest mainline analog:
- MLP output `c_proj`

Verdict:
- aligned with mainline

## Tied/global GDH params

These are tied across layers:

- `W_k_read_global`
- `W_v_read_global`
- `E_slots`
- `W_q_slots_global`
- `W_write_mlp_in_global` (if present)
- `W_write_mlp_out_global` (if present)

Mainline transformer blocks do not usually tie these types of matrices across layers, but tying does not change the **init family**. It only means layer 0's initialized tensor becomes the shared tensor.

## Matrix-by-matrix verdict table

| GDH param | Current init | Closest mainline analog | Mainline-like? | Note |
|---|---|---|---|---|
| `W_q_read` | `U(-s,s)` | `attn.c_q` | yes | good |
| `W_k_read_global` | `U(-s,s)` | `attn.c_k` | yes | good |
| `W_v_read_global` | `U(-s,s)` | `attn.c_v` | yes | good |
| `W_o_read` | `0` | `attn.c_proj` / `mlp.c_proj` | yes | good |
| `W_g_read_mute` | `0` | gate-like weight | ambiguous | VE gate uses small positive init |
| `b_g_read_mute` | `-1` | none | no | deliberate GDH-specific gate bias |
| `E_slots` | `U(-s,s)` | value embedding-like | yes-ish | reasonable keep |
| `W_q_slots_global` | `U(-s,s)` | `attn.c_q/c_k` | yes | good |
| `W_k_write` | `U(-s,s)` | `attn.c_k` | yes | good |
| `W_v_write` | `U(-s,s)` | `attn.c_v` | yes | good |
| `W_o_write` | `U(-s,s)` | `attn.c_proj` / `mlp.c_proj` | no | strict mainline says zero, but zero may kill GDH branch |
| `W_write_mlp_in_global` | `U(-s,s)` | `mlp.c_fc` | no | mainline uses `0.4x` scale |
| `W_write_mlp_out_global` | `0` | `mlp.c_proj` | yes | good |

## Clear differences worth discussing

These are the two cleanest init mismatches if the target is "make GDH matrices look like mainline matrices":

1. `W_write_mlp_in_global`
- current: `Uniform(-s, s)`
- mainline-like: `Uniform(-0.4s, 0.4s)`

2. `W_o_write`
- current: `Uniform(-s, s)`
- strict mainline-like: zeros
- but this likely conflicts with keeping GDH alive at initialization

Secondary / ambiguous differences:

3. `W_g_read_mute`
- current: zeros
- possible mainline-like gate analog: `Uniform(0.0, 0.02)`

4. `b_g_read_mute`
- current: `-1.0`
- mainline usually uses no bias here at all

## Practical recommendation before changing anything

If the goal is to move closer to mainline init style **without risking a dead GDH branch**, the safest first change is:

- change `W_write_mlp_in_global` to mainline MLP-input scaling:
  - `Uniform(-0.4s, 0.4s)`

I would be much more cautious about changing:

- `W_o_write -> 0`

because that is the one mismatch that is also plausibly carrying branch viability.
