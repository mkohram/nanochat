# Main Transformer Config for GDH/MQAR Probes

This note captures the **current main transformer/trunk configuration** used by the active `experiments/mqar_gdh_mps_lab.py` probe line.

It is specifically about the **main transformer** inherited from `nanochat/gpt.py`, not the GDH sidecar config.

## Source of truth

- Main trunk implementation:
  - `nanochat/gpt.py`
- Probe model construction:
  - `experiments/mqar_gdh_mps_lab.py`
- Active launchers:
  - `experiments/run_probe_easy_mps.sh`
  - `experiments/run_probe_hard_mps.sh`

## Main transformer config surface exposed by the probe

The probe currently passes these `GPTConfig` fields into the main transformer:

- `sequence_len`
- `vocab_size`
- `n_layer`
- `n_head`
- `n_kv_head`
- `n_embd`
- `window_pattern`

Current probe construction in `experiments/mqar_gdh_mps_lab.py` sets:

- `n_kv_head = n_head`
- `window_pattern = "L"`

So the active probe does **not** currently use reduced KV heads / GQA compression beyond standard equality with `n_head`.

## Active easy trunk config

From `experiments/run_probe_easy_mps.sh`:

- `sequence_len = 64`
- `vocab_size = 128`
- `n_layer = 3`
- `n_head = 4`
- `n_kv_head = 4`
- `n_embd = 64`
- `window_pattern = "L"`

Derived:

- `head_dim = n_embd / n_head = 16`

Blindfold behavior:

- launcher passes `--swa-window 8`
- probe wraps trunk in `WindowedGPT`
- effective attention window on **all layers** becomes:
  - `(8, 0)`

## Active hard trunk config

From `experiments/run_probe_hard_mps.sh`:

- `sequence_len = 256`
- `vocab_size = 128`
- `n_layer = 4`
- `n_head = 8`
- `n_kv_head = 8`
- `n_embd = 128`
- `window_pattern = "L"`

Derived:

- `head_dim = 128 / 8 = 16`

Blindfold behavior:

- launcher passes `--swa-window 8`
- effective attention window on **all layers** becomes:
  - `(8, 0)`

## Main transformer architectural behavior from `nanochat/gpt.py`

These are part of the effective trunk config even though they are not exposed as probe CLI flags.

### Core attention / block design

- rotary embeddings
- no learned positional embeddings
- causal self-attention
- QK norm enabled
- extra post-norm Q/K scaling:
  - `q *= 1.2`
  - `k *= 1.2`
- Flash Attention backend via `nanochat.flash_attention`
- no bias in linear layers
- custom `Linear` layer casts weights to activation dtype in forward
- MLP hidden width = `4 * n_embd`
- MLP activation = `relu(x)^2`
- token embeddings are normalized before entering blocks
- untied token embedding and LM head
- RMSNorm is functional; no learned norm parameters

### Block form

Each block is:

1. `x = x + attn(norm(x), ve, cos_sin, window_size, kv_cache)`
2. `x = x + mlp(norm(x))`

### Probe-used trunk tensors

The GDH lab directly uses these main-trunk components:

- `model.transformer.wte`
- `model.transformer.h`
- `model.resid_lambdas`
- `model.x0_lambdas`
- `model.value_embeds`
- `model.cos`
- `model.sin`
- `model.window_sizes`
- `model.lm_head`

## Value embedding configuration

Current mainline GPT keeps value embeddings enabled by default.

### Placement

Value embeddings exist on alternating layers, with the final layer always included.

Selector:

- `has_ve(layer_idx, n_layer)`

### Current VE path details

- VE gate input channels: `12`
- VE gate scale: `3 * sigmoid(...)`
- VE gate is per KV head

### Probe ablation switch

The lab now exposes:

- `--value-embeds`
- `--no-value-embeds`

Default:

- `--value-embeds` on

Implementation note:

- VE-off is implemented in the lab by replacing:
  - `model.value_embeds = nn.ModuleDict()`

## Attention window behavior

### Without blindfold

If `--swa-window 0`, the probe uses the trunk’s normal `window_pattern` logic.

Current probe sets:

- `window_pattern = "L"`

So without blindfold, all layers are full-context.

### With blindfold

If `--swa-window > 0`, the probe uses `WindowedGPT` to override all layers to the same sliding window.

Current active probes use:

- `--swa-window 8`

So all layers are forced to:

- `(8, 0)`

## Main trunk initialization behavior

Current `nanochat/gpt.py` init behavior:

### Embeddings / LM head

- token embedding:
  - `Normal(mean=0, std=0.8)`
- LM head:
  - `Normal(mean=0, std=0.001)`

### Attention matrices

Let:

- `s = sqrt(3) * n_embd^-0.5`

Then:

- `attn.c_q`: `Uniform(-s, s)`
- `attn.c_k`: `Uniform(-s, s)`
- `attn.c_v`: `Uniform(-s, s)`
- `attn.c_proj`: zeros

### MLP matrices

- `mlp.c_fc`: `Uniform(-0.4s, 0.4s)`
- `mlp.c_proj`: zeros

### Residual scalar params

- `resid_lambdas`: depth-varying init
- `x0_lambdas`: depth-varying init

These are **not** constant `1.0` / `0.1` anymore in current mainline GPT.

### Value embeddings / VE gate

- value embeddings: `Uniform(-s, s)`
- VE gate weights: `Uniform(0.0, 0.02)`

## Other mainline trunk parameters present in GPT

Current `nanochat/gpt.py` also defines:

- `smear_gate`
- `smear_lambda`
- `backout_lambda`

These are part of the mainline GPT object, but the current lab’s custom `_forward_gdh(...)` path does not explicitly use them.

## Runtime / dtype behavior for active MPS probes

From the launchers and current trunk/lab behavior:

- device: `mps`
- `--amp-dtype auto`
- `--compile` enabled
- trunk runs with current mainline `nanochat/gpt.py` dtype behavior
- GDH modules are intentionally kept in `fp32` on MPS in the lab to match `gdh-v0.1`
- GDH modules are cast to `bf16` only on CUDA

## Summary table

| Setting | Easy | Hard |
|---|---:|---:|
| `sequence_len` | 64 | 256 |
| `vocab_size` | 128 | 128 |
| `n_layer` | 3 | 4 |
| `n_head` | 4 | 8 |
| `n_kv_head` | 4 | 8 |
| `n_embd` | 64 | 128 |
| `head_dim` | 16 | 16 |
| `window_pattern` | `L` | `L` |
| `swa_window` | 8 | 8 |
| blindfolded | yes | yes |
| device | mps | mps |
| compile | on | on |
| value embeds default | on | on |

## Practical note

When interpreting probe regressions versus `gdh-v0.1`, remember that the main transformer trunk has changed in multiple meaningful ways since then, including:

- embedding init
- MLP init scale
- VE gate width/init/scale
- Q/K scaling
- rotary base
- residual scalar init
- custom `Linear` forward casting behavior

So probe differences are not attributable only to GDH-side changes.
