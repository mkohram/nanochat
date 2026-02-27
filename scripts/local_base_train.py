"""
Local/low-VRAM training entrypoint (GTX 1060-friendly).
From project root, run as:

python -m scripts.local_base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.local_base_train

Small test example:
python -m scripts.local_base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Temporary hard-disable compile path for GTX 1060 stability.
# Revert once architecture iteration is done.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["NANOCHAT_NO_COMPILE"] = "1"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig, norm as gpt_norm
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.recurrent_dataloader import tokenizing_distributed_data_loader_recurrent_doc, tokenizing_distributed_data_loader_with_state_recurrent_doc
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="L", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL'). Local default is L for SDPA/GTX stability.")
parser.add_argument("--force-swa-window", type=int, default=0, help="if >0, force this SWA window on ALL layers (including final layer); blindfold mode")
parser.add_argument("--arch", type=str, default="baseline", choices=["baseline", "gdh"], help="model architecture path")
parser.add_argument("--no-value-embeds", action="store_true", help="disable ResFormer-style value embeddings")
parser.add_argument("--gdh-slots", type=int, default=16, help="number of GDH sidecar slots (used when --arch gdh)")
parser.add_argument("--gdh-write-heads", type=int, default=-1, help="GDH write heads (-1 = use n_head)")
parser.add_argument("--gdh-no-read-gate", action="store_true", help="disable GDH read gate (always-on sidecar read path)")
parser.add_argument("--gdh-no-write-brain", action="store_true", help="disable GDH write-space residual brain (Linear->ReLU²->Linear)")
parser.add_argument("--gdh-write-brain-hidden-mult", type=int, default=4, help="GDH write-brain hidden width multiplier (hidden = mult * D)")
parser.add_argument("--gdh-route-topk", type=int, default=0, help="GDH sparse top-k write routing (0 = dense)")
parser.add_argument("--gdh-scan-beta", type=float, default=1.0, help="GDH temporal scan beta (1.0 = cumsum)")
parser.add_argument("--gdh-use-ema-scan", action="store_true", help="use GDH EMA scan S_t = g*S_{t-1} + (1-g)*delta_t with write-gate retention")
parser.add_argument("--gdh-usage-balance-lambda", type=float, default=0.0, help="GDH usage-balance auxiliary loss coefficient")
parser.add_argument("--gdh-use-write-gate", action="store_true", help="enable GDH write gate (per-layer Linear(D->1), sigmoid)")
parser.add_argument("--gdh-write-gate-bias", type=float, default=-2.0, help="initial bias for GDH write gate projections")
parser.add_argument("--gdh-probe-warmstart-read-mixer", action="store_true", help="warm-start first two GDH read output mixers (probe parity)")
parser.add_argument("--gate-log-every", type=int, default=10, help="log GDH gate telemetry every N steps (-1 = disable)")
parser.add_argument("--gate-open-threshold", type=float, default=0.55, help="threshold used for gate open fraction telemetry")
parser.add_argument("--train-log-every", type=int, default=10, help="log train metrics every N steps to align baseline/GDH telemetry density (-1 = disable)")
parser.add_argument("--gdh-out-hist-every", type=int, default=100, help="log GDH output-state histograms every N steps (-1 = disable)")
parser.add_argument("--gdh-weight-hist-every", type=int, default=100, help="log GDH read-weight histograms (W_o_read + shared read globals) every N steps (-1 = disable)")
parser.add_argument("--gdh-out-hist-max-points", type=int, default=4096, help="max tensor elements to include per histogram")
parser.add_argument("--freeze-local-path", action="store_true", help="freeze local backbone path (transformer blocks + value embeddings)")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
parser.add_argument("--resume-model-only", action="store_true", help="when resuming, load model/dataloader/step but reinitialize optimizer state")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=-1, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
parser.add_argument("--loader-mode", type=str, default="bos_bestfit", choices=["bos_bestfit", "recurrent_doc"], help="train/val dataloader mode")
parser.add_argument("--dataset", type=str, default="fineweb_edu", choices=["fineweb_edu", "tinystories"], help="text dataset source for local training")
parser.add_argument("--recurrent-max-chunks-per-doc", type=int, default=5, help="max contiguous chunks taken from one doc in recurrent_doc loader")
parser.add_argument("--gdh-recurrent-carry-state", action="store_true", help="carry GDH sidecar state across recurrent_doc chunks (detached TBPTT)")
args = parser.parse_args()

# Optional startup tracing to diagnose stalls in init path
TRACE_STARTUP = os.environ.get("NANOCHAT_STARTUP_TRACE", "0") in {"1", "true", "TRUE", "yes", "YES"}
_t0 = time.perf_counter()
def trace(msg):
    if TRACE_STARTUP:
        print(f"[startup +{time.perf_counter() - _t0:7.3f}s] {msg}", flush=True)

# Keep logged user config exactly as provided (no local architecture lock overrides).
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
trace(f"compute_init start (device_type={device_type})")
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
trace(f"compute_init done (ddp_world_size={ddp_world_size}, device={device})")
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
trace(f"wandb init start (dummy={use_dummy_wandb}, run={args.run})")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)
if not use_dummy_wandb:
    # Use explicit training step on the x-axis instead of wandb internal _step counter.
    # Keep the metric name as `step` for cleaner dashboards.
    wandb_run.define_metric("step")
    wandb_run.define_metric("*", step_metric="step")
trace("wandb init done")


def log_to_wandb(step_value, payload):
    wandb_run.log({
        "step": step_value,
        **payload,
    })

# Flash Attention status
if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
trace("tokenizer load start")
tokenizer = get_tokenizer()
trace("tokenizer load done")
trace("token bytes load start")
token_bytes = get_token_bytes(device=device)
trace("token bytes load done")
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
        force_swa_window=args.force_swa_window,
        arch=args.arch,
        gdh_slots=args.gdh_slots,
        gdh_write_heads=args.gdh_write_heads,
        gdh_use_read_gate=not args.gdh_no_read_gate,
        gdh_use_write_brain=not args.gdh_no_write_brain,
        gdh_write_brain_hidden_mult=args.gdh_write_brain_hidden_mult,
        gdh_route_topk=args.gdh_route_topk,
        gdh_scan_beta=args.gdh_scan_beta,
        gdh_use_ema_scan=args.gdh_use_ema_scan,
        gdh_usage_balance_lambda=args.gdh_usage_balance_lambda,
        gdh_use_write_gate=args.gdh_use_write_gate,
        gdh_write_gate_bias=args.gdh_write_gate_bias,
        gdh_probe_warmstart_read_mixer=args.gdh_probe_warmstart_read_mixer,
        gdh_bos_token_id=tokenizer.get_bos_token_id(),
        use_value_embeds=not args.no_value_embeds,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
trace(f"build_model_meta start (depth={args.depth})")
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
trace("build_model_meta done")
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
trace("model.to_empty start")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
trace("model.to_empty done")
trace("model.init_weights start")
model.init_weights() # 3) All tensors get initialized
trace("model.init_weights done")

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
default_tag = f"d{args.depth}" if args.arch == "baseline" else f"{args.arch}-d{args.depth}"
output_dirname = args.model_tag if args.model_tag else default_tag # e.g. d12, gdh-d1
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    if args.resume_model_only:
        print0(f"Resuming model/dataloader state from step {args.resume_from_step} (optimizer reinitialized)")
    else:
        print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=not args.resume_model_only,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

if args.freeze_local_path:
    frozen_params = 0
    for p in model.transformer.h.parameters():
        p.requires_grad_(False)
        frozen_params += p.numel()
    for p in model.value_embeds.parameters():
        p.requires_grad_(False)
        frozen_params += p.numel()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(
        f"Freezing local path (transformer.h + value_embeds): {frozen_params:,} params frozen; "
        f"{trainable_params:,} params remain trainable"
    )

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: only convert layers with dimensions divisible by 16 (FP8 hardware requirement)
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            # FP8 requires both in_features and out_features divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers, skipped {num_skipped} (dims not divisible by 16)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model (optional)

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
if os.environ.get("NANOCHAT_NO_COMPILE", "0") in {"1", "true", "TRUE", "yes", "YES"}:
    print0("Skipping torch.compile because NANOCHAT_NO_COMPILE is enabled")
else:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
trace("optimizer setup start")
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)
trace("optimizer setup done")

if resuming and not args.resume_model_only:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
trace("train dataloader build start")
if args.loader_mode == "bos_bestfit":
    if args.dataset != "fineweb_edu":
        raise ValueError("bos_bestfit currently supports only --dataset fineweb_edu")
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
    )
elif args.loader_mode == "recurrent_doc":
    train_loader = tokenizing_distributed_data_loader_with_state_recurrent_doc(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
        resume_state_dict=dataloader_resume_state_dict,
        max_chunks_per_doc=args.recurrent_max_chunks_per_doc,
        source_dataset=args.dataset,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_recurrent_doc(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
        max_chunks_per_doc=args.recurrent_max_chunks_per_doc,
        source_dataset=args.dataset,
    )
else:
    raise ValueError(f"Unknown loader mode: {args.loader_mode}")
trace("first batch prefetch start")
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data
trace("first batch prefetch done")

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# GDH gate telemetry helper

def _to_wandb_histogram(t: torch.Tensor, max_points: int):
    flat = t.detach().float().reshape(-1)
    if flat.numel() == 0:
        return wandb.Histogram([0.0])
    # Guard against NaN/Inf so telemetry never crashes training.
    finite = torch.isfinite(flat)
    if not finite.any():
        return wandb.Histogram([0.0])
    flat = flat[finite]
    if flat.numel() > max_points:
        stride = max(flat.numel() // max_points, 1)
        flat = flat[::stride][:max_points]
    return wandb.Histogram(flat.cpu().tolist())


def collect_gdh_probe_telemetry(
    m: GPT,
    idx_tokens: torch.Tensor,
    open_threshold: float,
    *,
    include_out_hist: bool,
    include_weight_hist: bool,
    out_hist_max_points: int,
):
    """Collect GDH probe telemetry from one token batch.

    Includes:
    - gate stats
    - output-state (sidecar) stats
    - optional output-state histograms
    - optional read-weight histograms (W_o_read + shared read globals)
    """
    if getattr(m, "arch", "baseline") != "gdh":
        return {}

    with torch.no_grad():
        autocast_gate = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if idx_tokens.device.type == "cuda" else nullcontext()
        with autocast_gate:
            bsz, n_tokens = idx_tokens.shape
            x = m.transformer.wte(idx_tokens)
            x = gpt_norm(x)
            x0 = x

            gdh_slots = m.config.gdh_slots
            gdh_heads = m.config.n_head if m.config.gdh_write_heads <= 0 else m.config.gdh_write_heads
            sidecar = torch.zeros(bsz, n_tokens, gdh_slots, m.config.n_embd, device=x.device, dtype=x.dtype)
            cos_sin = m.cos[:, :n_tokens], m.sin[:, :n_tokens]
            gdh_boundary_mask = (idx_tokens == int(getattr(m.config, "gdh_bos_token_id", 1))) if bool(getattr(m.config, "gdh_segmented_scan", True)) else None

            telemetry = {}
            gate_means, gate_stds, gate_open_fracs = [], [], []
            out_means, out_abs_means, out_stds = [], [], []
            delta_abs_means = []
            use_read_gate = bool(getattr(m.config, "gdh_use_read_gate", True))
            use_write_brain = bool(getattr(m.config, "gdh_use_write_brain", True))

            for i, block in enumerate(m.transformer.h):
                x = m.resid_lambdas[i] * x + m.x0_lambdas[i] * x0

                ve = m.value_embeds[str(i)](idx_tokens) if str(i) in m.value_embeds else None
                x = m.gdh_read[i].forward_sequence(x, sidecar, eps=1e-6)
                x = block(x, ve, cos_sin, m.window_sizes[i], kv_cache=None)

                gate_proj = m.gdh_write_gate[i] if m.gdh_write_gate is not None else None
                delta, _, g_write = m._gdh_write_probe_delta(
                    m.gdh_write[i],
                    x,
                    gdh_heads=gdh_heads,
                    route_topk=getattr(m.config, "gdh_route_topk", 0),
                    gate_proj=gate_proj,
                    eps=1e-6,
                )

                if g_write is not None:
                    g = g_write.squeeze(-1).squeeze(-1).float()
                else:
                    g = torch.ones(bsz, n_tokens, device=x.device, dtype=torch.float32)

                gate_mean = g.mean().item()
                gate_std = g.std(unbiased=False).item()
                gate_open = (g > open_threshold).float().mean().item()
                gate_means.append(gate_mean)
                gate_stds.append(gate_std)
                gate_open_fracs.append(gate_open)

                delta = torch.tanh(delta)
                if bool(getattr(m.config, "gdh_use_ema_scan", False)) and g_write is not None:
                    sidecar = m._gdh_scan_accumulate_ema(
                        delta,
                        retention=g_write,
                        boundary_mask=gdh_boundary_mask,
                    )
                else:
                    sidecar = sidecar + m._gdh_scan_accumulate(
                        delta,
                        beta=float(getattr(m.config, "gdh_scan_beta", 1.0)),
                        boundary_mask=gdh_boundary_mask,
                    )

                delta_f = delta.float()
                sidecar_f = sidecar.float()

                out_mean = sidecar_f.mean().item()
                out_abs_mean = sidecar_f.abs().mean().item()
                out_std = sidecar_f.std(unbiased=False).item()
                delta_abs_mean = delta_f.abs().mean().item()

                out_means.append(out_mean)
                out_abs_means.append(out_abs_mean)
                out_stds.append(out_std)
                delta_abs_means.append(delta_abs_mean)

                telemetry[f"gdh/layer_{i}/write_gate_mean"] = gate_mean
                telemetry[f"gdh/layer_{i}/write_gate_std"] = gate_std
                telemetry[f"gdh/layer_{i}/write_gate_open_frac"] = gate_open

                telemetry[f"gdh/layer_{i}/out_state_mean"] = out_mean
                telemetry[f"gdh/layer_{i}/out_state_abs_mean"] = out_abs_mean
                telemetry[f"gdh/layer_{i}/out_state_std"] = out_std
                telemetry[f"gdh/layer_{i}/delta_abs_mean"] = delta_abs_mean
                telemetry[f"gdh/layer_{i}/out_state_final_token_abs_mean"] = sidecar_f[:, -1].abs().mean().item()

                if include_out_hist:
                    telemetry[f"gdh/layer_{i}/out_state_hist"] = _to_wandb_histogram(sidecar_f, out_hist_max_points)

                if include_weight_hist:
                    w_o_read_f = m.gdh_read[i].W_o_read.float()
                    telemetry[f"gdh/layer_{i}/W_o_read_abs_mean"] = w_o_read_f.abs().mean().item()
                    telemetry[f"gdh/layer_{i}/W_o_read_std"] = w_o_read_f.std(unbiased=False).item()
                    telemetry[f"gdh/layer_{i}/W_o_read_hist"] = _to_wandb_histogram(w_o_read_f, out_hist_max_points)

    if include_weight_hist and len(m.gdh_read) > 0:
        read0 = m.gdh_read[0]
        w_k_read_global = read0.W_k_read_global.float()
        w_v_read_global = read0.W_v_read_global.float()

        telemetry["gdh/global/W_k_read_global_abs_mean"] = w_k_read_global.abs().mean().item()
        telemetry["gdh/global/W_k_read_global_std"] = w_k_read_global.std(unbiased=False).item()
        telemetry["gdh/global/W_k_read_global_hist"] = _to_wandb_histogram(w_k_read_global, out_hist_max_points)

        telemetry["gdh/global/W_v_read_global_abs_mean"] = w_v_read_global.abs().mean().item()
        telemetry["gdh/global/W_v_read_global_std"] = w_v_read_global.std(unbiased=False).item()
        telemetry["gdh/global/W_v_read_global_hist"] = _to_wandb_histogram(w_v_read_global, out_hist_max_points)

        write0 = m.gdh_write[0]
        e_slots = write0.E_slots.float()
        w_q_slots_global = write0.W_q_slots_global.float()

        telemetry["gdh/global/E_slots_abs_mean"] = e_slots.abs().mean().item()
        telemetry["gdh/global/E_slots_std"] = e_slots.std(unbiased=False).item()
        telemetry["gdh/global/E_slots_hist"] = _to_wandb_histogram(e_slots, out_hist_max_points)

        telemetry["gdh/global/W_q_slots_global_abs_mean"] = w_q_slots_global.abs().mean().item()
        telemetry["gdh/global/W_q_slots_global_std"] = w_q_slots_global.std(unbiased=False).item()
        telemetry["gdh/global/W_q_slots_global_hist"] = _to_wandb_histogram(w_q_slots_global, out_hist_max_points)

        if getattr(write0, "W_write_mlp_in_global", None) is not None:
            w_write_mlp_in = write0.W_write_mlp_in_global.float()
            w_write_mlp_out = write0.W_write_mlp_out_global.float()
            telemetry["gdh/global/W_write_mlp_in_abs_mean"] = w_write_mlp_in.abs().mean().item()
            telemetry["gdh/global/W_write_mlp_in_std"] = w_write_mlp_in.std(unbiased=False).item()
            telemetry["gdh/global/W_write_mlp_in_hist"] = _to_wandb_histogram(w_write_mlp_in, out_hist_max_points)
            telemetry["gdh/global/W_write_mlp_out_abs_mean"] = w_write_mlp_out.abs().mean().item()
            telemetry["gdh/global/W_write_mlp_out_std"] = w_write_mlp_out.std(unbiased=False).item()
            telemetry["gdh/global/W_write_mlp_out_hist"] = _to_wandb_histogram(w_write_mlp_out, out_hist_max_points)

        if len(m.gdh_read) > 1:
            k_tied = all(
                m.gdh_read[j].W_k_read_global.data_ptr() == read0.W_k_read_global.data_ptr()
                for j in range(1, len(m.gdh_read))
            )
            v_tied = all(
                m.gdh_read[j].W_v_read_global.data_ptr() == read0.W_v_read_global.data_ptr()
                for j in range(1, len(m.gdh_read))
            )
            e_slots_tied = all(
                m.gdh_write[j].E_slots.data_ptr() == write0.E_slots.data_ptr()
                for j in range(1, len(m.gdh_write))
            )
            q_slots_tied = all(
                m.gdh_write[j].W_q_slots_global.data_ptr() == write0.W_q_slots_global.data_ptr()
                for j in range(1, len(m.gdh_write))
            )

            telemetry["gdh/global/W_k_read_global_tied"] = float(k_tied)
            telemetry["gdh/global/W_v_read_global_tied"] = float(v_tied)
            telemetry["gdh/global/E_slots_tied"] = float(e_slots_tied)
            telemetry["gdh/global/W_q_slots_global_tied"] = float(q_slots_tied)

            if getattr(write0, "W_write_mlp_in_global", None) is not None:
                write_mlp_in_tied = all(
                    m.gdh_write[j].W_write_mlp_in_global.data_ptr() == write0.W_write_mlp_in_global.data_ptr()
                    for j in range(1, len(m.gdh_write))
                )
                write_mlp_out_tied = all(
                    m.gdh_write[j].W_write_mlp_out_global.data_ptr() == write0.W_write_mlp_out_global.data_ptr()
                    for j in range(1, len(m.gdh_write))
                )
                telemetry["gdh/global/W_write_mlp_in_tied"] = float(write_mlp_in_tied)
                telemetry["gdh/global/W_write_mlp_out_tied"] = float(write_mlp_out_tied)

    telemetry.update({
        "gdh/use_read_gate": float(use_read_gate),
        "gdh/use_write_brain": float(use_write_brain),
        "gdh/write_brain_hidden_mult": float(getattr(m.config, "gdh_write_brain_hidden_mult", 0)),
        "gdh/write_gate_mean": sum(gate_means) / len(gate_means),
        "gdh/write_gate_std": sum(gate_stds) / len(gate_stds),
        "gdh/write_gate_open_frac": sum(gate_open_fracs) / len(gate_open_fracs),
        "gdh/out_state_mean": sum(out_means) / len(out_means),
        "gdh/out_state_abs_mean": sum(out_abs_means) / len(out_abs_means),
        "gdh/out_state_std": sum(out_stds) / len(out_stds),
        "gdh/delta_abs_mean": sum(delta_abs_means) / len(delta_abs_means),
    })
    return telemetry


# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Optional recurrent sidecar carry (GDH only)
use_gdh_recurrent_carry = (
    args.arch == "gdh"
    and args.loader_mode == "recurrent_doc"
    and args.gdh_recurrent_carry_state
)
gdh_state_carry = None
chunk_loss_sum = {}
chunk_loss_count = {}

# Go!
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or (step > 0 and step % args.eval_every == 0)):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        log_to_wandb(step, {
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        log_to_wandb(step, {
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    current_chunk_idx = None
    for micro_step in range(grad_accum_steps):
        if args.loader_mode == "recurrent_doc":
            current_chunk_idx = int(dataloader_state_dict.get("chunk_idx_in_doc", -1))
        with autocast_ctx:
            if use_gdh_recurrent_carry:
                loss, gdh_state_next = model(
                    x,
                    y,
                    gdh_state_in=gdh_state_carry,
                    return_gdh_state=True,
                )
            else:
                loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        if use_gdh_recurrent_carry:
            # Legacy carry decay (0.85) for additive scan. For EMA scan, recurrence already applies
            # per-token retention and bounded updates, so keep carry unscaled.
            if bool(getattr(model.config, "gdh_use_ema_scan", False)):
                gdh_state_carry = gdh_state_next.detach()
            else:
                gdh_state_carry = gdh_state_next.detach() * 0.85
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    if args.loader_mode == "recurrent_doc" and current_chunk_idx is not None and current_chunk_idx >= 0:
        chunk_loss_sum[current_chunk_idx] = chunk_loss_sum.get(current_chunk_idx, 0.0) + train_loss_f
        chunk_loss_count[current_chunk_idx] = chunk_loss_count.get(current_chunk_idx, 0) + 1
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    gate_log_due = args.arch == "gdh" and args.gate_log_every > 0 and (step % args.gate_log_every == 0)
    out_hist_due = args.arch == "gdh" and args.gdh_out_hist_every > 0 and (step % args.gdh_out_hist_every == 0)
    weight_hist_due = args.arch == "gdh" and args.gdh_weight_hist_every > 0 and (step % args.gdh_weight_hist_every == 0)
    train_log_due = args.train_log_every > 0 and (step % args.train_log_every == 0)
    gdh_probe_due = args.arch == "gdh" and (gate_log_due or out_hist_due or weight_hist_due)
    gdh_probe_telemetry = {}

    if gdh_probe_due:
        gdh_probe_telemetry = collect_gdh_probe_telemetry(
            orig_model,
            x,
            args.gate_open_threshold,
            include_out_hist=out_hist_due,
            include_weight_hist=weight_hist_due,
            out_hist_max_points=args.gdh_out_hist_max_points,
        )
        if gdh_probe_telemetry:
            if gate_log_due:
                print0(
                    f"gdh write_gate | mean: {gdh_probe_telemetry['gdh/write_gate_mean']:.4f} | "
                    f"std: {gdh_probe_telemetry['gdh/write_gate_std']:.4f} | "
                    f"open>{args.gate_open_threshold:.2f}: {gdh_probe_telemetry['gdh/write_gate_open_frac']:.3f}"
                )
            print0(
                f"gdh out_state | mean: {gdh_probe_telemetry['gdh/out_state_mean']:.6f} | "
                f"abs_mean: {gdh_probe_telemetry['gdh/out_state_abs_mean']:.6f} | "
                f"std: {gdh_probe_telemetry['gdh/out_state_std']:.6f}"
            )

    # If GDH probe is due but train logging isn't, emit a telemetry-only point.
    # If both are due, emit a single combined point in the train log below.
    if gdh_probe_telemetry and not train_log_due:
        log_to_wandb(step, {
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            **gdh_probe_telemetry,
        })

    if train_log_due:
        log_data = {
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
            **gdh_probe_telemetry,
        }
        if args.loader_mode == "recurrent_doc" and current_chunk_idx is not None and current_chunk_idx >= 0:
            log_data["train/chunk_id_current"] = float(current_chunk_idx)
            log_data["train/chunk_loss_current"] = train_loss_f
            for k in sorted(chunk_loss_sum.keys()):
                c = chunk_loss_count.get(k, 0)
                if c > 0:
                    log_data[f"train/chunk_loss_avg_{k}"] = chunk_loss_sum[k] / c
            if 0 in chunk_loss_sum and current_chunk_idx in chunk_loss_sum and current_chunk_idx != 0:
                c0 = chunk_loss_sum[0] / max(chunk_loss_count.get(0, 1), 1)
                ck = chunk_loss_sum[current_chunk_idx] / max(chunk_loss_count.get(current_chunk_idx, 1), 1)
                log_data["train/chunk_loss_gap_vs_0"] = ck - c0
        log_to_wandb(step, log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
