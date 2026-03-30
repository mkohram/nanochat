#!/usr/bin/env python3
"""
Minimal MQAR lab harness for first-principles GDH debugging.

Purpose:
- isolate the reduced GDH core rather than mirror mainline exactly
- study the simplest write->accumulate->read loop that still exposes the
  main pathologies we care about:
  - state growth / blow-up
  - slot collapse
  - stable learning vs stalled learning
  - capacity stress behavior

This file is intentionally narrower than mainline GDH. It is a lab bench,
not a production-parity harness.

Included mechanisms:
- baseline vs GDH
- additive scan / leaky scan beta sweep
- optional sparse top-k routing
- optional routing-usage balancing loss
- blindfolded SWA evaluation mode
- answer-only MQAR metrics plus a small set of state diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, norm as gpt_norm


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_autocast(device: str, amp_dtype: str):
    if amp_dtype == "off":
        return nullcontext()
    if device == "cuda":
        dtype = torch.bfloat16 if amp_dtype == "auto" else getattr(torch, amp_dtype)
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    if device == "mps":
        # Default MPS probes to bf16; in this lab it is behaving more stably than fp16.
        dtype = torch.bfloat16 if amp_dtype == "auto" else getattr(torch, amp_dtype)
        return torch.amp.autocast(device_type="mps", dtype=dtype)
    return nullcontext()


class WindowedGPT(GPT):
    """GPT subclass with forced sliding-window attention on all layers."""

    def __init__(self, config: GPTConfig, swa_window: int = 0):
        # Must set this before super().__init__ because _compute_window_sizes runs in init.
        self._swa_window = swa_window
        super().__init__(config)

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        if self._swa_window <= 0:
            return super()._compute_window_sizes(config)

        # Blindfold mode: force the same sliding window on every layer,
        # including the final layer, so attention cannot cheat across the gap.
        window = (self._swa_window, 0)
        return [window] * config.n_layer


class LabGDHReadCore(nn.Module):
    """Minimal GDH read core kept local to the lab harness."""

    def __init__(self, d: int, *, use_read_mute_gate: bool = True):
        super().__init__()
        self.use_read_mute_gate = use_read_mute_gate
        self.W_q_read = nn.Parameter(torch.empty(d, d))
        self.W_k_read_global = nn.Parameter(torch.empty(d, d))
        self.W_v_read_global = nn.Parameter(torch.empty(d, d))
        self.W_o_read = nn.Parameter(torch.empty(d, d))
        self.W_g_read_mute = nn.Parameter(torch.empty(d, 1))
        self.b_g_read_mute = nn.Parameter(torch.zeros(1))

    def reset_parameters(self, *, std: float, zero_init_mixer: bool) -> None:
        del zero_init_mixer
        s = math.sqrt(3.0) * std
        nn.init.uniform_(self.W_q_read, -s, s)
        nn.init.uniform_(self.W_k_read_global, -s, s)
        nn.init.uniform_(self.W_v_read_global, -s, s)
        nn.init.zeros_(self.W_g_read_mute)
        nn.init.constant_(self.b_g_read_mute, -1.0)
        nn.init.zeros_(self.W_o_read)


class LabGDHWriteCore(nn.Module):
    """Minimal GDH write core kept local to the lab harness."""

    def __init__(
        self,
        d: int,
        r: int,
        *,
        use_write_brain: bool = False,
        write_brain_hidden_mult: int = 4,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("LabGDHWriteCore requires positive slot count r")
        self.E_slots = nn.Parameter(torch.empty(r, d))
        self.W_q_slots_global = nn.Parameter(torch.empty(d, d))
        self.W_k_write = nn.Parameter(torch.empty(d, d))
        self.W_v_write = nn.Parameter(torch.empty(d, d))
        self.W_o_write = nn.Parameter(torch.empty(d, d))
        self.use_write_brain = use_write_brain
        self.write_brain_hidden_mult = write_brain_hidden_mult
        if use_write_brain:
            hidden = write_brain_hidden_mult * d
            self.W_write_mlp_in_global = nn.Parameter(torch.empty(d, hidden))
            self.W_write_mlp_out_global = nn.Parameter(torch.empty(hidden, d))
        else:
            self.register_parameter("W_write_mlp_in_global", None)
            self.register_parameter("W_write_mlp_out_global", None)

    def reset_parameters(self, *, std: float, zero_init_mixer: bool) -> None:
        del zero_init_mixer
        s = math.sqrt(3.0) * std
        nn.init.uniform_(self.E_slots, -s, s)
        nn.init.uniform_(self.W_q_slots_global, -s, s)
        nn.init.uniform_(self.W_k_write, -s, s)
        nn.init.uniform_(self.W_v_write, -s, s)
        nn.init.uniform_(self.W_o_write, -s, s)
        if self.use_write_brain and self.W_write_mlp_in_global is not None:
            nn.init.uniform_(self.W_write_mlp_in_global, -0.4 * s, 0.4 * s)
            nn.init.zeros_(self.W_write_mlp_out_global)


def _tie_gdh_global_weights_local(read_cores: nn.ModuleList, write_cores: nn.ModuleList) -> None:
    if len(read_cores) <= 1:
        return

    shared_k_read = read_cores[0].W_k_read_global
    shared_v_read = read_cores[0].W_v_read_global
    shared_e_slots = write_cores[0].E_slots
    shared_q_slots = write_cores[0].W_q_slots_global
    shared_write_mlp_in = write_cores[0].W_write_mlp_in_global
    shared_write_mlp_out = write_cores[0].W_write_mlp_out_global

    for read_core in read_cores[1:]:
        read_core.W_k_read_global = shared_k_read
        read_core.W_v_read_global = shared_v_read
    for write_core in write_cores[1:]:
        write_core.E_slots = shared_e_slots
        write_core.W_q_slots_global = shared_q_slots
        if shared_write_mlp_in is not None:
            write_core.W_write_mlp_in_global = shared_write_mlp_in
            write_core.W_write_mlp_out_global = shared_write_mlp_out


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _scan_accumulate(delta: torch.Tensor, scan_beta: float) -> torch.Tensor:
    """Token-wise accumulation for the reduced lab probe.

    - beta=1.0: pure cumsum
    - 0<beta<1: leaky scan y_t = beta*y_{t-1} + delta_t
    - beta<=0: no carry, only current delta

    Vectorized closed form:
      y_t = beta^t * cumsum(delta_t * beta^{-t})

    Use fp32 for the internal leaky-scan math so this stays on-device on MPS
    (float64 is unsupported there) while retaining better numerical stability
    than bf16/fp16 autocast inputs.
    """
    if scan_beta >= 1.0:
        return torch.cumsum(delta, dim=1)
    if scan_beta <= 0.0:
        return delta

    t = delta.shape[1]
    work_dtype = torch.float32
    beta = torch.tensor(float(scan_beta), device=delta.device, dtype=work_dtype)
    idx = torch.arange(t, device=delta.device, dtype=work_dtype)
    beta_pow = torch.pow(beta, idx)
    beta_inv = torch.pow(beta, -idx)

    x = delta.to(work_dtype)
    y = torch.cumsum(x * beta_inv.view(1, t, 1, 1), dim=1) * beta_pow.view(1, t, 1, 1)
    return y.to(delta.dtype)


def _slot_stats_last(sidecar: torch.Tensor, eps: float = 1e-9) -> dict[str, float]:
    """Final-token slot geometry / usage stats, averaged across batch."""
    s = sidecar.float()[:, -1]  # [B,R,D]
    norms = s.norm(dim=-1)  # [B,R]
    s_unit = s / norms.unsqueeze(-1).clamp_min(eps)
    gram = torch.einsum("brd,bsd->brs", s_unit, s_unit)  # [B,R,R]

    r = gram.shape[-1]
    eye = torch.eye(r, dtype=torch.bool, device=gram.device)
    offdiag = gram[:, ~eye]
    offdiag_cpu = offdiag.to(torch.float32).cpu()

    usage = norms / norms.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(usage.clamp_min(eps) * usage.clamp_min(eps).log()).sum(dim=-1)
    effective_slots = entropy.exp()
    max_share = usage.max(dim=-1).values

    norm_sq_sum = norms.pow(2).sum(dim=-1)
    norm_four_sum = norms.pow(4).sum(dim=-1).clamp_min(eps)
    participation_ratio = norm_sq_sum.pow(2) / norm_four_sum

    norm_mean = norms.mean(dim=-1)
    norm_std = norms.std(dim=-1, unbiased=False)
    norm_cv = norm_std / norm_mean.clamp_min(eps)

    offdiag_p90 = torch.quantile(offdiag_cpu, 0.90, dim=-1).mean()

    vals = [
        float(offdiag.mean().cpu()),
        float(offdiag.max(dim=-1).values.mean().cpu()),
        float(offdiag_p90),
        float(effective_slots.mean().cpu()),
        float(max_share.mean().cpu()),
        float(participation_ratio.mean().cpu()),
        float(norm_mean.mean().cpu()),
        float(norm_cv.mean().cpu()),
    ]

    return {
        "slot_cos_mean": float(vals[0]),
        "slot_cos_max": float(vals[1]),
        "slot_cos_p90": float(vals[2]),
        "effective_slots": float(vals[3]),
        "max_share": float(vals[4]),
        "participation_ratio": float(vals[5]),
        "slot_norm_mean": float(vals[6]),
        "slot_norm_cv": float(vals[7]),
    }


def _out_state_hist_last(sidecar: torch.Tensor, bins: int = 80) -> dict[str, Any]:
    """Adaptive histogram + stats for final-token sidecar values (no clipping)."""
    x = sidecar.float()[:, -1].reshape(-1)
    if x.numel() == 0:
        return {"bins": [], "values": [], "stats": {}}

    x_min_t, x_max_t = torch.aminmax(x)
    x_min, x_max = torch.stack([x_min_t, x_max_t]).cpu().tolist()
    if x_max <= x_min:
        eps = 1e-6
        x_min -= eps
        x_max += eps

    vals = torch.histc(x, bins=bins, min=x_min, max=x_max).cpu().tolist()
    edges = torch.linspace(x_min, x_max, bins + 1, device=x.device).cpu().tolist()

    q = torch.quantile(x, torch.tensor([0.01, 0.50, 0.99], device=x.device, dtype=x.dtype))
    stat_vals = torch.cat([
        torch.stack([x.mean(), x.std(unbiased=False)]),
        q,
        x.abs().max().unsqueeze(0),
    ]).cpu().tolist()
    stats = {
        "min": float(x_min),
        "max": float(x_max),
        "mean": float(stat_vals[0]),
        "std": float(stat_vals[1]),
        "p01": float(stat_vals[2]),
        "p50": float(stat_vals[3]),
        "p99": float(stat_vals[4]),
        "abs_max": float(stat_vals[5]),
    }
    return {"bins": edges, "values": vals, "stats": stats}


def _compute_answer_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eval_topk: int,
    *,
    compute_topk: bool = True,
    compute_mrr: bool = True,
) -> dict[str, float | None]:
    """Compute answer-only metrics using masked targets (-100 excluded)."""
    with torch.no_grad():
        valid = targets.ne(-100)
        n_valid_t = valid.sum()
        n_valid = int(n_valid_t.detach().cpu())
        topk_key = f"acc_top{eval_topk}"
        if n_valid == 0:
            return {
                "acc_top1": 0.0,
                topk_key: 0.0 if compute_topk else None,
                "mrr": 0.0 if compute_mrr else None,
                "n_answers": 0.0,
            }

        sel_logits = logits[valid]
        sel_targets = targets[valid].long()

        pred1 = sel_logits.argmax(dim=-1)
        acc1_t = (pred1 == sel_targets).float().mean()

        stats = [acc1_t, n_valid_t.float()]
        if compute_topk:
            k = max(1, min(eval_topk, sel_logits.shape[-1]))
            topk = torch.topk(sel_logits, k=k, dim=-1).indices
            hitk = (topk == sel_targets.unsqueeze(1)).any(dim=1)
            stats.append(hitk.float().mean())
        else:
            k = max(1, min(eval_topk, sel_logits.shape[-1]))

        if compute_mrr:
            target_scores = sel_logits.gather(1, sel_targets.unsqueeze(1))
            ranks = (sel_logits > target_scores).sum(dim=1) + 1
            stats.append((1.0 / ranks.float()).mean())

        stat_vals = torch.stack(stats).cpu().tolist()
        idx = 0
        acc1 = float(stat_vals[idx]); idx += 1
        n_answers = float(stat_vals[idx]); idx += 1
        acck = float(stat_vals[idx]) if compute_topk else None
        idx += 1 if compute_topk else 0
        mrr = float(stat_vals[idx]) if compute_mrr else None

        return {
            "acc_top1": acc1,
            f"acc_top{k}": acck,
            "mrr": mrr,
            "n_answers": n_answers,
        }


def _read_from_sidecar(
    read_core,
    local: torch.Tensor,
    sidecar_prev: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Reduced GDH read path used by this lab harness.

    This is intentionally implemented locally in the probe so GDH-side math lives in
    `experiments/mqar_gdh_mps_lab.py` rather than delegating to `nanochat.double_helix`.

    Current probe ablation default:
    - read-mute branch exists locally, but can be disabled with
      `read_core.use_read_mute_gate = False`
    """
    bsz, n_tokens, d = local.shape
    assert sidecar_prev.shape[:2] == (bsz, n_tokens)

    x_read = _rms_norm(local, eps=eps)                                  # [B,N,D]
    q_loc = torch.matmul(x_read, read_core.W_q_read)                    # [B,N,D]

    s_hat = _rms_norm(sidecar_prev, eps=eps)                            # [B,N,R,D]
    k_mem = torch.einsum("bnrd,df->bnrf", s_hat, read_core.W_k_read_global)
    v_mem = torch.einsum("bnrd,df->bnrf", s_hat, read_core.W_v_read_global)

    logits = torch.einsum("bnd,bnrd->bnr", q_loc, k_mem) / math.sqrt(d)
    alpha = torch.softmax(logits, dim=-1)                               # [B,N,R]
    z_read = torch.einsum("bnr,bnrd->bnd", alpha, v_mem)               # [B,N,D]

    z_proj = torch.matmul(z_read, read_core.W_o_read)                   # [B,N,D]
    if getattr(read_core, "use_read_mute_gate", True):
        read_mute = torch.sigmoid(torch.matmul(x_read, read_core.W_g_read_mute) + read_core.b_g_read_mute)
        z_proj = read_mute * z_proj
    return local + z_proj


def _apply_write_brain_local(write_core, delta: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Probe-local write-brain residual, kept here so GDH math stays self-contained."""
    if not getattr(write_core, "use_write_brain", False) or write_core.W_write_mlp_in_global is None:
        return delta
    delta_norm = _rms_norm(delta, eps=eps)
    hidden = torch.relu(torch.matmul(delta_norm, write_core.W_write_mlp_in_global)).square()
    return delta + torch.matmul(hidden, write_core.W_write_mlp_out_global)


def _write_delta(
    write_core,
    local_out: torch.Tensor,
    sidecar_prev: torch.Tensor,
    *,
    n_write_heads: int,
    route_topk: int,
    routing_mode: str,
    eps: float,
) -> torch.Tensor:
    """Reduced GDH write path used by this lab harness.

    Routing ablations:
      - static: learned slot addresses only
      - content: current sidecar contents only
      - hybrid: static + content logits

    Returns:
      delta: [B,N,R,D]
    """
    bsz, n_tokens, d = local_out.shape
    assert sidecar_prev.shape[:2] == (bsz, n_tokens)
    r = sidecar_prev.shape[2]
    assert write_core.E_slots.shape[0] == r, "sidecar slot count must match E_slots"
    d_h = d // n_write_heads

    x_write = _rms_norm(local_out, eps=eps)
    k_upd = torch.matmul(x_write, write_core.W_k_write)
    v_upd = torch.matmul(x_write, write_core.W_v_write)

    k_h = k_upd.view(bsz, n_tokens, n_write_heads, d_h)
    v_h = v_upd.view(bsz, n_tokens, n_write_heads, d_h)

    logits = None

    if routing_mode in {"static", "hybrid"}:
        e_slots = _rms_norm(write_core.E_slots, eps=eps)
        q_slots_static = torch.matmul(e_slots, write_core.W_q_slots_global)
        q_static_h = q_slots_static.view(r, n_write_heads, d_h)
        logits_static = torch.einsum("rhd,bnhd->bnhr", q_static_h, k_h) / math.sqrt(d_h)
        logits = logits_static if logits is None else logits + logits_static

    if routing_mode in {"content", "hybrid"}:
        s_slots = _rms_norm(sidecar_prev, eps=eps)
        q_slots_content = torch.einsum("bnrd,df->bnrf", s_slots, write_core.W_q_slots_global)
        q_content_h = q_slots_content.view(bsz, n_tokens, r, n_write_heads, d_h)
        logits_content = torch.einsum("bnrhd,bnhd->bnhr", q_content_h, k_h) / math.sqrt(d_h)
        logits = logits_content if logits is None else logits + logits_content

    if logits is None:
        raise ValueError(f"unknown routing_mode: {routing_mode}")

    alpha = torch.softmax(logits, dim=-1)

    if route_topk > 0 and route_topk < r:
        top_idx = torch.topk(alpha, k=route_topk, dim=-1).indices
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, top_idx, 1.0)
        alpha = alpha * mask
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    delta_heads = torch.einsum("bnhr,bnhd->bnhrd", alpha, v_h)
    delta_raw = delta_heads.permute(0, 1, 3, 2, 4).contiguous().view(bsz, n_tokens, r, d)
    delta = torch.einsum("bnrd,df->bnrf", delta_raw, write_core.W_o_write)
    delta = _apply_write_brain_local(write_core, delta, eps=eps)

    return delta


def _make_run_label(args: argparse.Namespace, beta: float) -> str:
    return (
        f"arch={args.arch}"
        f"__beta={beta:g}"
        f"__topk={args.route_topk}"
        f"__wroute={args.write_routing}"
        f"__rmg={int(bool(args.read_mute_gate))}"
        f"__wb={int(bool(args.gdh_use_write_brain))}"
        f"__ve={int(bool(args.value_embeds))}"
        f"__slots={args.gdh_slots}"
        f"__pairs={args.n_pairs}"
        f"__queries={args.n_queries}"
        f"__gap={args.gap_min}-{args.gap_max}"
        f"__seed={args.seed}"
    )


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(vars(args))
    return cfg


def _build_model(args: argparse.Namespace, device: str) -> GPT:
    torch.manual_seed(args.seed)

    if args.gdh_write_heads <= 0:
        gdh_heads = args.n_head
    else:
        gdh_heads = args.gdh_write_heads
    if args.n_embd % gdh_heads != 0:
        raise ValueError("n_embd must be divisible by gdh_write_heads (or n_head if gdh_write_heads<=0)")

    cfg = GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
        window_pattern="L",
    )
    cfg.arch = args.arch
    cfg.gdh_slots = args.gdh_slots
    cfg.gdh_write_heads = args.gdh_write_heads
    cfg.gdh_use_write_brain = args.gdh_use_write_brain
    cfg.gdh_write_brain_hidden_mult = args.gdh_write_brain_hidden_mult

    if args.swa_window > 0:
        model = WindowedGPT(cfg, swa_window=args.swa_window).to(device)
    else:
        model = GPT(cfg).to(device)

    model.init_weights()
    model.arch = args.arch

    if not args.value_embeds:
        model.value_embeds = nn.ModuleDict()

    if args.arch == "gdh":
        model.gdh_read = nn.ModuleList([
            LabGDHReadCore(model.config.n_embd, use_read_mute_gate=bool(args.read_mute_gate))
            for _ in range(model.config.n_layer)
        ]).to(device)
        model.gdh_write = nn.ModuleList([
            LabGDHWriteCore(
                model.config.n_embd,
                args.gdh_slots,
                use_write_brain=bool(args.gdh_use_write_brain),
                write_brain_hidden_mult=args.gdh_write_brain_hidden_mult,
            )
            for _ in range(model.config.n_layer)
        ]).to(device)
        _tie_gdh_global_weights_local(model.gdh_read, model.gdh_write)

        gdh_std = model.config.n_embd ** -0.5
        for read_core, write_core in zip(model.gdh_read, model.gdh_write):
            read_core.reset_parameters(std=gdh_std, zero_init_mixer=False)
            write_core.reset_parameters(std=gdh_std, zero_init_mixer=False)

        if model.transformer.wte.weight.device.type == "cuda":
            model.gdh_read.to(dtype=torch.bfloat16)
            model.gdh_write.to(dtype=torch.bfloat16)

        with torch.no_grad():
            for read_core in model.gdh_read:
                read_core.use_read_mute_gate = bool(args.read_mute_gate)
    else:
        model.gdh_read = None
        model.gdh_write = None

    return model


def make_mqar_batch(args: argparse.Namespace, batch_size: int, device: str):
    """Create MQAR batch with masked targets only at answer positions.

    Full sequence pattern (length = sequence_len + 1):
      [K1 V1 K2 V2 ... KP VP] [filler gap] [Q1 A1 Q2 A2 ... QQ AQ] [tail filler]

    Training objective is next-token prediction only at positions where current token is Q*.
    """

    n_full = args.sequence_len + 1
    kv_len = 2 * args.n_pairs
    qa_len = 2 * args.n_queries

    if kv_len + qa_len >= n_full:
        raise ValueError("sequence too short for configured n_pairs/n_queries")

    key_vocab = args.key_vocab
    value_offset = args.key_vocab
    value_vocab = args.value_vocab
    query_offset = args.query_offset
    filler_offset = args.filler_offset
    filler_vocab = args.filler_vocab

    full = torch.randint(
        filler_offset,
        filler_offset + filler_vocab,
        (batch_size, n_full),
        dtype=torch.long,
        device=device,
    )

    key_weights = torch.ones(batch_size, key_vocab, device=device)
    value_weights = torch.ones(batch_size, value_vocab, device=device)
    pair_weights = torch.ones(batch_size, args.n_pairs, device=device)

    keys = torch.multinomial(key_weights, num_samples=args.n_pairs, replacement=False)
    vals = value_offset + torch.multinomial(value_weights, num_samples=args.n_pairs, replacement=False)

    kv_key_pos = torch.arange(0, kv_len, 2, device=device).unsqueeze(0).expand(batch_size, -1)
    kv_val_pos = kv_key_pos + 1
    full.scatter_(1, kv_key_pos, keys)
    full.scatter_(1, kv_val_pos, vals)

    max_gap = n_full - kv_len - qa_len
    if max_gap < 0:
        raise ValueError("Invalid layout; reduce n_pairs/n_queries or increase sequence_len")

    gap_lo = min(args.gap_min, max_gap)
    gap_hi = min(args.gap_max, max_gap)
    if gap_hi < gap_lo:
        gap = torch.full((batch_size,), max_gap, dtype=torch.long, device=device)
    else:
        gap = torch.randint(gap_lo, gap_hi + 1, (batch_size,), dtype=torch.long, device=device)

    q_start = kv_len + gap
    q_idx = torch.multinomial(pair_weights, num_samples=args.n_queries, replacement=False)
    q_keys = keys.gather(1, q_idx)
    q_vals = vals.gather(1, q_idx)

    q_offsets = 2 * torch.arange(args.n_queries, device=device, dtype=torch.long).unsqueeze(0)
    q_pos = q_start.unsqueeze(1) + q_offsets
    a_pos = q_pos + 1

    full.scatter_(1, q_pos, query_offset + q_keys)
    full.scatter_(1, a_pos, q_vals)

    idx = full[:, :-1]
    nxt = full[:, 1:]

    targets = torch.full_like(nxt, -100)
    is_query = (idx >= query_offset) & (idx < query_offset + key_vocab)
    targets[is_query] = nxt[is_query]

    return idx, targets


def _forward_baseline(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    eval_topk: int,
    compute_topk: bool,
    compute_mrr: bool,
):
    logits = model(idx)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    metrics = _compute_answer_metrics(logits, targets, eval_topk=eval_topk, compute_topk=compute_topk, compute_mrr=compute_mrr)
    return ce_loss, ce_loss, metrics, None


def _forward_gdh(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    scan_beta: float,
    route_topk: int,
    write_routing: str,
    eval_topk: int,
    collect_sidecar: bool,
    compute_topk: bool,
    compute_mrr: bool,
):
    bsz, n_tokens = idx.size()

    x = model.transformer.wte(idx)
    x = gpt_norm(x)
    x0 = x

    gdh_slots = model.config.gdh_slots
    gdh_heads = model.config.n_head if model.config.gdh_write_heads <= 0 else model.config.gdh_write_heads

    sidecar = torch.zeros(bsz, n_tokens, gdh_slots, model.config.n_embd, device=x.device, dtype=x.dtype)
    cos_sin = model.cos[:, :n_tokens], model.sin[:, :n_tokens]

    layer_slot_cos = []
    layer_slot_cos_max = []
    layer_slot_cos_p90 = []
    layer_effective_slots = []
    layer_max_share = []
    layer_participation_ratio = []
    layer_slot_norm_mean = []
    layer_slot_norm_cv = []
    layer_sidecar_norm_trace = []
    layer_sidecar_last = []
    layer_out_state_hist = []

    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        x = _read_from_sidecar(model.gdh_read[i], x, sidecar, eps=1e-6)
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache=None)

        delta = _write_delta(
            model.gdh_write[i],
            x,
            sidecar,
            n_write_heads=gdh_heads,
            route_topk=route_topk,
            routing_mode=write_routing,
            eps=1e-6,
        )

        sidecar = sidecar + _scan_accumulate(delta, scan_beta=scan_beta)

        if collect_sidecar:
            slot_stats = _slot_stats_last(sidecar)
            layer_slot_cos.append(slot_stats["slot_cos_mean"])
            layer_slot_cos_max.append(slot_stats["slot_cos_max"])
            layer_slot_cos_p90.append(slot_stats["slot_cos_p90"])
            layer_effective_slots.append(slot_stats["effective_slots"])
            layer_max_share.append(slot_stats["max_share"])
            layer_participation_ratio.append(slot_stats["participation_ratio"])
            layer_slot_norm_mean.append(slot_stats["slot_norm_mean"])
            layer_slot_norm_cv.append(slot_stats["slot_norm_cv"])
            layer_sidecar_norm_trace.append(sidecar[0].float().norm(dim=-1).cpu().tolist())
            layer_sidecar_last.append(sidecar[0, -1].float().cpu().tolist())
            layer_out_state_hist.append(_out_state_hist_last(sidecar))

    x = gpt_norm(x)
    logits = model.lm_head(x)[..., :model.config.vocab_size].float()
    softcap = 15
    logits = softcap * torch.tanh(logits / softcap)

    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    total_loss = ce_loss

    metrics = _compute_answer_metrics(logits, targets, eval_topk=eval_topk, compute_topk=compute_topk, compute_mrr=compute_mrr)
    if collect_sidecar:
        stats = {
            "slot_cos_mean_last_per_layer": layer_slot_cos,
            "slot_cos_max_last_per_layer": layer_slot_cos_max,
            "slot_cos_p90_last_per_layer": layer_slot_cos_p90,
            "slot_usage_effective_slots_last_per_layer": layer_effective_slots,
            "slot_usage_max_share_last_per_layer": layer_max_share,
            "slot_participation_ratio_last_per_layer": layer_participation_ratio,
            "slot_norm_mean_last_per_layer": layer_slot_norm_mean,
            "slot_norm_cv_last_per_layer": layer_slot_norm_cv,
            "sidecar_norm_trace_sample0_last_per_layer": layer_sidecar_norm_trace,
            "sidecar_last_sample0_last_per_layer": layer_sidecar_last,
            "out_state_hist_last_per_layer": layer_out_state_hist,
        }
    else:
        stats = None
    return total_loss, ce_loss, metrics, stats


def run_one_beta(args: argparse.Namespace, beta: float, device: str) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = _build_model(args, device=device)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    model.train()

    run_label = _make_run_label(args, beta)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else args.steps
    if lr_decay_iters <= 0:
        raise ValueError("lr_decay_iters must be > 0")
    base_lr = float(args.lr)
    min_lr = float(args.min_lr)
    if min_lr < 0.0:
        raise ValueError("min_lr must be >= 0")
    if min_lr > base_lr:
        raise ValueError("min_lr must be <= lr")

    def get_lr(step_idx: int) -> float:
        if step_idx >= lr_decay_iters:
            return min_lr
        ratio = float(step_idx) / float(lr_decay_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (base_lr - min_lr)

    print(
        f"[{run_label}] lr_schedule base={base_lr:.6g} min={min_lr:.6g} decay_iters={lr_decay_iters}",
        flush=True,
    )

    ac = _make_autocast(device, args.amp_dtype)

    torch.manual_seed(args.seed + 1)
    eval_idx, eval_tgt = make_mqar_batch(args, batch_size=args.eval_batch_size, device=device)

    history: list[dict[str, Any]] = []
    run_start_time = time.perf_counter()

    def eval_record(
        step: int,
        train_total: float | None = None,
        train_ce: float | None = None,
        train_acc1: float | None = None,
        train_mrr: float | None = None,
    ):
        full_metrics = (
            step == 0
            or step == args.steps
            or (args.full_metrics_every > 0 and step % args.full_metrics_every == 0)
        )
        model.eval()
        with torch.no_grad():
            with ac:
                if args.arch == "baseline":
                    eval_total, eval_ce, eval_metrics, stats = _forward_baseline(
                        model,
                        eval_idx,
                        eval_tgt,
                        eval_topk=args.eval_topk,
                        compute_topk=full_metrics,
                        compute_mrr=full_metrics,
                    )
                else:
                    eval_total, eval_ce, eval_metrics, stats = _forward_gdh(
                        model,
                        eval_idx,
                        eval_tgt,
                        scan_beta=beta,
                        route_topk=args.route_topk,
                        write_routing=args.write_routing,
                        eval_topk=args.eval_topk,
                        collect_sidecar=True,
                        compute_topk=full_metrics,
                        compute_mrr=full_metrics,
                    )

        wall_time_s = time.perf_counter() - run_start_time
        eval_scalars = torch.stack([eval_total.detach(), eval_ce.detach()]).cpu().tolist()
        rec: dict[str, Any] = {
            "step": step,
            "beta": beta,
            "run_label": run_label,
            "lr": float(opt.param_groups[0]["lr"]),
            "wall_time_s": float(wall_time_s),
            "wall_time_min": float(wall_time_s / 60.0),
            "eval_total": float(eval_scalars[0]),
            "eval_ce": float(eval_scalars[1]),
            "eval_acc_top1": float(eval_metrics["acc_top1"]),
            "eval_mrr": None if eval_metrics["mrr"] is None else float(eval_metrics["mrr"]),
            "eval_n_answers": float(eval_metrics["n_answers"]),
            "eval_acc": float(eval_metrics["acc_top1"]),
            "full_metrics": bool(full_metrics),
        }
        topk_key = f"acc_top{max(1, min(args.eval_topk, args.vocab_size))}"
        rec[f"eval_{topk_key}"] = None if eval_metrics[topk_key] is None else float(eval_metrics[topk_key])

        if args.arch == "gdh":
            rec["slot_cos_l_last"] = float(stats["slot_cos_mean_last_per_layer"][-1])
            rec["slot_cos_layers"] = [float(v) for v in stats["slot_cos_mean_last_per_layer"]]
            rec["slot_cos_max_layers"] = [float(v) for v in stats["slot_cos_max_last_per_layer"]]
            rec["slot_cos_p90_layers"] = [float(v) for v in stats["slot_cos_p90_last_per_layer"]]
            rec["slot_usage_effective_slots_layers"] = [float(v) for v in stats["slot_usage_effective_slots_last_per_layer"]]
            rec["slot_usage_max_share_layers"] = [float(v) for v in stats["slot_usage_max_share_last_per_layer"]]
            rec["slot_participation_ratio_layers"] = [float(v) for v in stats["slot_participation_ratio_last_per_layer"]]
            rec["slot_norm_mean_layers"] = [float(v) for v in stats["slot_norm_mean_last_per_layer"]]
            rec["slot_norm_cv_layers"] = [float(v) for v in stats["slot_norm_cv_last_per_layer"]]
            rec["sidecar_norm_trace_layers"] = stats.get("sidecar_norm_trace_sample0_last_per_layer", [])
            rec["sidecar_last_layers"] = stats.get("sidecar_last_sample0_last_per_layer", [])
            rec["out_state_hist_layers"] = stats.get("out_state_hist_last_per_layer", [])
            rec["out_state_hist"] = rec["out_state_hist_layers"][-1] if rec["out_state_hist_layers"] else None
            rec["out_state_stats_layers"] = [h.get("stats", {}) for h in rec["out_state_hist_layers"] if isinstance(h, dict)]
            rec["out_state_stats"] = rec["out_state_stats_layers"][-1] if rec["out_state_stats_layers"] else None
        else:
            rec["slot_cos_l_last"] = None
            rec["slot_cos_layers"] = []
            rec["slot_cos_max_layers"] = []
            rec["slot_cos_p90_layers"] = []
            rec["slot_usage_effective_slots_layers"] = []
            rec["slot_usage_max_share_layers"] = []
            rec["slot_participation_ratio_layers"] = []
            rec["slot_norm_mean_layers"] = []
            rec["slot_norm_cv_layers"] = []
            rec["sidecar_norm_trace_layers"] = []
            rec["sidecar_last_layers"] = []
            rec["out_state_hist_layers"] = []
            rec["out_state_hist"] = None
            rec["out_state_stats_layers"] = []
            rec["out_state_stats"] = None

        if train_total is not None:
            rec["train_total"] = float(train_total)
        if train_ce is not None:
            rec["train_ce"] = float(train_ce)
        if train_acc1 is not None:
            rec["train_acc_top1"] = float(train_acc1)
        if train_mrr is not None:
            rec["train_mrr"] = float(train_mrr)

        history.append(rec)

        if getattr(args, "live_json", ""):
            try:
                live_payload = {
                    "config": _config_payload(args),
                    "beta": beta,
                    "run_label": run_label,
                    "history": history,
                    "last": rec,
                }
                tmp_path = args.live_json + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(live_payload, f)
                os.replace(tmp_path, args.live_json)
            except Exception:
                pass

        cos_txt = "n/a" if rec["slot_cos_l_last"] is None else f"{rec['slot_cos_l_last']:.4f}"
        mrr_txt = "n/a" if rec["eval_mrr"] is None else f"{rec['eval_mrr']:.4f}"
        print(
            f"[{run_label}] step={step} eval_acc1={rec['eval_acc_top1']:.4f} "
            f"eval_mrr={mrr_txt} eval_ce={rec['eval_ce']:.4f} "
            f"slot_cos={cos_txt} lr={rec['lr']:.6g}",
            flush=True,
        )

        model.train()

    eval_record(step=0)

    for step in range(1, args.steps + 1):
        step_idx = step - 1
        lr_t = get_lr(step_idx)
        for group in opt.param_groups:
            group["lr"] = lr_t

        idx, tgt = make_mqar_batch(args, batch_size=args.batch_size, device=device)

        opt.zero_grad(set_to_none=True)
        with ac:
            if args.arch == "baseline":
                total, ce, train_metrics, _ = _forward_baseline(
                    model,
                    idx,
                    tgt,
                    eval_topk=args.eval_topk,
                    compute_topk=False,
                    compute_mrr=False,
                )
            else:
                total, ce, train_metrics, _ = _forward_gdh(
                    model,
                    idx,
                    tgt,
                    scan_beta=beta,
                    route_topk=args.route_topk,
                    write_routing=args.write_routing,
                    eval_topk=args.eval_topk,
                    collect_sidecar=False,
                    compute_topk=False,
                    compute_mrr=False,
                )
        total.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            eval_record(
                step=step,
                train_total=float(total.item()),
                train_ce=float(ce.item()),
                train_acc1=float(train_metrics["acc_top1"]),
                train_mrr=None if train_metrics["mrr"] is None else float(train_metrics["mrr"]),
            )

    first = history[0]
    last = history[-1]
    return {
        "beta": beta,
        "run_label": run_label,
        "first": first,
        "last": last,
        "history": history,
    }


def maybe_plot(out_json: str, payload: dict[str, Any]) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        runs = payload["runs"]
        has_slot = any(r["last"].get("slot_cos_l_last") is not None for r in runs)

        if has_slot:
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=170, sharex=True)
            ax1, ax2, ax3, ax4 = axes
        else:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=170, sharex=True)
            ax1, ax2, ax3 = axes
            ax4 = None

        for run in runs:
            beta = run["beta"]
            hist = run["history"]
            steps = [h["step"] for h in hist]

            ax1.plot(steps, [h["eval_acc_top1"] for h in hist], marker="o", label=f"beta={beta}")
            ax2.plot(steps, [h["eval_mrr"] for h in hist], marker="o", label=f"beta={beta}")
            ax3.plot(steps, [h["eval_ce"] for h in hist], marker="o", label=f"beta={beta}")

            if ax4 is not None:
                ys = [float("nan") if h["slot_cos_l_last"] is None else h["slot_cos_l_last"] for h in hist]
                ax4.plot(steps, ys, marker="o", label=f"beta={beta}")

        ax1.set_ylabel("MQAR top1")
        ax1.grid(alpha=0.25)
        ax1.legend(loc="best", fontsize=8)

        ax2.set_ylabel("MQAR MRR")
        ax2.grid(alpha=0.25)

        ax3.set_ylabel("MQAR masked CE")
        ax3.grid(alpha=0.25)

        if ax4 is not None:
            ax4.set_ylabel("last-layer slot cosine")
            ax4.grid(alpha=0.25)
            ax4.set_xlabel("step")
        else:
            ax3.set_xlabel("step")

        fig.suptitle(f"MQAR GDH lab ({payload['config']['arch']})", y=0.995)
        fig.tight_layout()

        out_png = out_json.replace(".json", ".png")
        fig.savefig(out_png)
        return out_png
    except Exception:
        return None


def parse_betas(text: str) -> list[float]:
    vals = []
    for chunk in text.split(","):
        c = chunk.strip()
        if not c:
            continue
        vals.append(float(c))
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal MQAR GDH lab harness (baseline/GDH)")

    p.add_argument("--arch", type=str, default="gdh", choices=["baseline", "gdh"])
    p.add_argument("--betas", type=str, default="1.0,0.99,0.95,0.9")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--sequence-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=128)

    p.add_argument("--gdh-slots", type=int, default=8)
    p.add_argument("--gdh-write-heads", type=int, default=8)
    p.add_argument(
        "--gdh-use-write-brain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable probe-local write-brain residual MLP (default: off)",
    )
    p.add_argument("--gdh-write-brain-hidden-mult", type=int, default=4)
    p.add_argument("--route-topk", type=int, default=0)
    p.add_argument("--write-routing", type=str, default="static", choices=["static", "content", "hybrid"], help="write routing source: learned slot addresses, current sidecar contents, or both")
    p.add_argument(
        "--read-mute-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable probe-local read-mute gate (default: off)",
    )
    p.add_argument(
        "--value-embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable transformer value embeddings from the mainline GPT trunk (default: on)",
    )

    p.add_argument("--swa-window", type=int, default=0, help="Sliding window size (0=off). If set, forces SWA on all layers.")

    p.add_argument("--n-pairs", type=int, default=16)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--gap-min", type=int, default=64)
    p.add_argument("--gap-max", type=int, default=192)

    p.add_argument("--key-vocab", type=int, default=2048)
    p.add_argument("--value-vocab", type=int, default=2048)
    p.add_argument("--query-offset", type=int, default=4096)
    p.add_argument("--filler-offset", type=int, default=6144)
    p.add_argument("--filler-vocab", type=int, default=2048)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-topk", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--full-metrics-every", type=int, default=200, help="compute expensive eval top-k + MRR every N eval points; step 0 and final step always compute full metrics")
    p.add_argument("--lr-decay-iters", type=int, default=-1, help="cosine LR decay horizon in optimizer steps (-1 => use --steps)")
    p.add_argument("--min-lr", type=float, default=0.0, help="minimum LR at/after decay horizon")
    p.add_argument("--enforce-capacity-stress", action="store_true", help="require n_pairs > gdh_slots")

    p.add_argument("--device", type=str, default=_default_device(), choices=["cuda", "mps", "cpu"])
    p.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "off", "float16", "bfloat16"], help="autocast dtype policy; auto=bf16 on cuda, bf16 on mps, off on cpu")
    p.add_argument("--compile", action="store_true", help="compile the model with torch.compile (useful for longer fixed-shape probe runs)")
    p.add_argument("--live-json", type=str, default="", help="optional path to continuously write live metrics JSON")
    args = p.parse_args()

    if args.n_queries > args.n_pairs:
        raise ValueError("n_queries must be <= n_pairs")
    if args.query_offset + args.key_vocab > args.vocab_size:
        raise ValueError("query token range exceeds vocab_size")
    if args.filler_offset + args.filler_vocab > args.vocab_size:
        raise ValueError("filler token range exceeds vocab_size")
    if args.arch == "gdh" and args.enforce_capacity_stress and args.n_pairs <= args.gdh_slots:
        raise ValueError("capacity stress requested but n_pairs <= gdh_slots")
    if args.swa_window > 0 and args.gap_min <= args.swa_window:
        raise ValueError(f"gap_min ({args.gap_min}) must be > swa_window ({args.swa_window}) to ensure blindfold")

    if args.arch == "baseline" and args.route_topk != 0:
        print("[warn] route_topk ignored for baseline arch", flush=True)

    betas = parse_betas(args.betas)
    if args.arch == "baseline":
        betas = [betas[0]]

    runs = []
    for beta in betas:
        runs.append(run_one_beta(args, beta=beta, device=args.device))

    topk_key = f"acc_top{max(1, min(args.eval_topk, args.vocab_size))}"

    summary_rows = []
    for r in runs:
        summary_rows.append({
            "run_label": r["run_label"],
            "beta": r["beta"],
            "eval_acc_top1_0": r["first"]["eval_acc_top1"],
            "eval_acc_top1_last": r["last"]["eval_acc_top1"],
            f"eval_{topk_key}_last": r["last"][f"eval_{topk_key}"],
            "eval_mrr_last": r["last"]["eval_mrr"],
            "eval_ce_last": r["last"]["eval_ce"],
            "slot_cos_last": r["last"]["slot_cos_l_last"],
        })

    payload = {
        "config": _config_payload(args),
        "betas": betas,
        "summary": summary_rows,
        "runs": runs,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"mqar_gdh_mps_lab_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    out_png = maybe_plot(out_json, payload)

    print(json.dumps({
        "out_json": out_json,
        "out_png": out_png,
        "summary": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
