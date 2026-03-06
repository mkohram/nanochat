#!/usr/bin/env python3
"""
MQAR (Multi-Query Associative Recall) probe for GDH vs baseline.

Supports:
- Curriculum-friendly configs (easy Stage-0 to hard stress tests)
- Optional leaky scan beta sweep (GDH)
- Optional sparse top-k routing (GDH)
- Optional routing-usage balancing loss (GDH)
- Metrics beyond top-1: top-k and MRR

Testbed-only: does not modify main training path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, norm as gpt_norm


class WindowedGPT(GPT):
    """GPT subclass with forced sliding window attention on all layers."""
    def __init__(self, config: GPTConfig, swa_window: int = 0):
        # Must set this before super().__init__ because _compute_window_sizes runs in init
        self._swa_window = swa_window
        super().__init__(config)

    def _compute_window_sizes(self, config: GPTConfig) -> list[tuple[int, int]]:
        if self._swa_window <= 0:
            return super()._compute_window_sizes(config)
        
        # Override all layers (including final!) to use the specified sliding window.
        # This is CRITICAL for the "Blindfold" test: if final layer is full context,
        # it can cheat across the gap.
        window = (self._swa_window, 0)
        return [window] * config.n_layer


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _scan_accumulate(delta: torch.Tensor, scan_beta: float) -> torch.Tensor:
    """Token-wise accumulation.

    - beta=1.0: pure cumsum
    - 0<beta<1: leaky scan y_t = beta*y_{t-1} + delta_t

    Vectorized closed form:
      y_t = beta^t * cumsum(delta_t * beta^{-t})
    """
    if scan_beta >= 1.0:
        return torch.cumsum(delta, dim=1)
    if scan_beta <= 0.0:
        return delta

    t = delta.shape[1]
    beta = torch.tensor(float(scan_beta), device=delta.device, dtype=torch.float64)
    idx = torch.arange(t, device=delta.device, dtype=torch.float64)
    beta_pow = torch.pow(beta, idx)
    beta_inv = torch.pow(beta, -idx)

    x = delta.to(torch.float64)
    y = torch.cumsum(x * beta_inv.view(1, t, 1, 1), dim=1) * beta_pow.view(1, t, 1, 1)
    return y.to(delta.dtype)


def _pairwise_slot_cos_last_mean(sidecar: torch.Tensor, eps: float = 1e-9) -> float:
    """Mean pairwise slot cosine at final token, averaged across batch."""
    s = sidecar.float()[:, -1]  # [B,R,D]
    s = s / s.norm(dim=-1, keepdim=True).clamp_min(eps)
    gram = torch.einsum("brd,bsd->brs", s, s)  # [B,R,R]

    r = gram.shape[-1]
    mask = ~torch.eye(r, dtype=torch.bool, device=gram.device)
    vals = gram[:, mask]
    return float(vals.mean().item())


def _compute_answer_metrics(logits: torch.Tensor, targets: torch.Tensor, eval_topk: int) -> dict[str, float]:
    """Compute answer-only metrics using masked targets (-100 excluded)."""
    with torch.no_grad():
        valid = targets.ne(-100)
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            return {
                "acc_top1": 0.0,
                f"acc_top{eval_topk}": 0.0,
                "mrr": 0.0,
                "n_answers": 0.0,
            }

        sel_logits = logits[valid]           # [M,V]
        sel_targets = targets[valid].long()  # [M]

        # top-1
        pred1 = sel_logits.argmax(dim=-1)
        acc1 = float((pred1 == sel_targets).float().mean().item())

        # top-k
        k = max(1, min(eval_topk, sel_logits.shape[-1]))
        topk = torch.topk(sel_logits, k=k, dim=-1).indices
        hitk = (topk == sel_targets.unsqueeze(1)).any(dim=1)
        acck = float(hitk.float().mean().item())

        # MRR: rank = 1 + count(scores > target_score)
        target_scores = sel_logits.gather(1, sel_targets.unsqueeze(1))
        ranks = (sel_logits > target_scores).sum(dim=1) + 1
        mrr = float((1.0 / ranks.float()).mean().item())

        return {
            "acc_top1": acc1,
            f"acc_top{k}": acck,
            "mrr": mrr,
            "n_answers": float(n_valid),
        }


def _write_delta_and_alpha(
    write_core,
    local_out: torch.Tensor,
    sidecar_prev: torch.Tensor,
    *,
    n_write_heads: int,
    route_topk: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Mirror mainline GDH write delta with optional top-k routing.

    Returns:
      delta: [B,N,R,D]
      alpha: [B,N,h,R] (hard/masked)
      alpha_soft: [B,N,h,R] (soft/unmasked)
      g: [B,N,1,1] (gate, or None)
    """
    bsz, n_tokens, d = local_out.shape
    r = sidecar_prev.shape[2]
    d_h = d // n_write_heads

    x_write = _rms_norm(local_out, eps=eps)
    k_upd = torch.matmul(x_write, write_core.W_k_write)
    v_upd = torch.matmul(x_write, write_core.W_v_write)

    e_slots = _rms_norm(write_core.E_slots, eps=eps)
    q_slots = torch.matmul(e_slots, write_core.W_q_slots_global)

    q_h = q_slots.view(r, n_write_heads, d_h)
    k_h = k_upd.view(bsz, n_tokens, n_write_heads, d_h)
    v_h = v_upd.view(bsz, n_tokens, n_write_heads, d_h)

    logits = torch.einsum("rhd,bnhd->bnhr", q_h, k_h) / math.sqrt(d_h)
    alpha_soft = torch.softmax(logits, dim=-1)
    alpha = alpha_soft.clone()

    if route_topk > 0 and route_topk < r:
        top_idx = torch.topk(alpha, k=route_topk, dim=-1).indices
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, top_idx, 1.0)
        alpha = alpha * mask
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    delta_heads = torch.einsum("bnhr,bnhd->bnhrd", alpha, v_h)
    delta_raw = delta_heads.permute(0, 1, 3, 2, 4).contiguous().view(bsz, n_tokens, r, d)
    delta = torch.einsum("bnrd,df->bnrf", delta_raw, write_core.W_o_write)
    delta = write_core._apply_write_brain(delta, eps=eps)

    g_write = None
    return delta, alpha, alpha_soft, g_write


def _apply_legacy_20260223_gdh_freeze(model: GPT) -> None:
    """Force GDH probe internals toward 2026-02-23 probe-era behavior.

    This intentionally decouples probe behavior from evolving mainline defaults.
    """
    if getattr(model, "arch", "").lower() != "gdh":
        return

    std = model.config.n_embd ** -0.5
    with torch.no_grad():
        for i in range(len(model.gdh_read)):
            r = model.gdh_read[i]
            w = model.gdh_write[i]

            # Legacy read path: no read-mute gate in probe-era runs.
            if hasattr(r, "use_read_mute_gate"):
                r.use_read_mute_gate = False

            # Legacy-ish matrix init family used in probe-era GDH modules.
            torch.nn.init.normal_(r.W_q_read, mean=0.0, std=std)
            torch.nn.init.normal_(r.W_k_read_global, mean=0.0, std=std)
            torch.nn.init.normal_(r.W_v_read_global, mean=0.0, std=std)
            torch.nn.init.normal_(r.W_o_read, mean=0.0, std=std)

            torch.nn.init.normal_(w.E_slots, mean=0.0, std=1.0)
            torch.nn.init.normal_(w.W_q_slots_global, mean=0.0, std=std)
            torch.nn.init.normal_(w.W_k_write, mean=0.0, std=std)
            torch.nn.init.normal_(w.W_v_write, mean=0.0, std=std)
            torch.nn.init.normal_(w.W_o_write, mean=0.0, std=std)

            if getattr(w, "W_write_mlp_in_global", None) is not None:
                torch.nn.init.normal_(w.W_write_mlp_in_global, mean=0.0, std=std)
                torch.nn.init.zeros_(w.W_write_mlp_out_global)

        # Probe-era warmstart for first read mixers.
        for i in range(min(2, len(model.gdh_read))):
            torch.nn.init.normal_(model.gdh_read[i].W_o_read, mean=0.0, std=0.02)


def _build_model(args: argparse.Namespace, device: str) -> GPT:
    torch.manual_seed(args.seed)

    cfg = GPTConfig(
        sequence_len=args.sequence_len,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
        window_pattern="L",
        arch=args.arch,
        gdh_slots=args.gdh_slots,
        gdh_write_heads=args.gdh_write_heads,
        gdh_use_write_brain=True,
        gdh_write_brain_hidden_mult=4,
    )

    if args.swa_window > 0:
        model = WindowedGPT(cfg, swa_window=args.swa_window).to(device)
    else:
        model = GPT(cfg).to(device)

    model.init_weights()

    if args.arch == "gdh":
        _apply_legacy_20260223_gdh_freeze(model)

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

    full = torch.empty(batch_size, n_full, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Unique keys and unique values within sample.
        keys = torch.randperm(key_vocab, device=device)[: args.n_pairs]
        vals = value_offset + torch.randperm(value_vocab, device=device)[: args.n_pairs]

        # Fill with random filler tokens first.
        seq = torch.randint(filler_offset, filler_offset + filler_vocab, (n_full,), device=device)

        # KV write phase.
        pos = 0
        for i in range(args.n_pairs):
            seq[pos] = keys[i]
            seq[pos + 1] = vals[i]
            pos += 2

        # Variable filler gap before queries.
        max_gap = n_full - pos - qa_len
        if max_gap < 0:
            raise ValueError("Invalid layout; reduce n_pairs/n_queries or increase sequence_len")

        gap_lo = min(args.gap_min, max_gap)
        gap_hi = min(args.gap_max, max_gap)
        if gap_hi < gap_lo:
            gap = max_gap
        else:
            gap = int(torch.randint(gap_lo, gap_hi + 1, (1,), device=device).item())

        q_start = pos + gap

        # Query phase: sample seen keys.
        q_idx = torch.randperm(args.n_pairs, device=device)[: args.n_queries]
        for qi in range(args.n_queries):
            ki = int(q_idx[qi].item())
            k = int(keys[ki].item())
            v = int(vals[ki].item())
            seq[q_start + 2 * qi] = query_offset + k
            seq[q_start + 2 * qi + 1] = v

        full[b] = seq

    # Next-token setup.
    idx = full[:, :-1]
    nxt = full[:, 1:]

    # Mask loss to query->answer edges only.
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
):
    logits = model(idx)
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    metrics = _compute_answer_metrics(logits, targets, eval_topk=eval_topk)
    zero = ce_loss.new_zeros(())
    return ce_loss, ce_loss, zero, metrics, None


def _forward_gdh(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    scan_beta: float,
    route_topk: int,
    usage_balance_lambda: float,
    eval_topk: int,
    collect_sidecar: bool,
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
    usage_losses = []

    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        x = model.gdh_read[i].forward_sequence(x, sidecar, eps=1e-6)
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache=None)

        delta, alpha, alpha_soft, g = _write_delta_and_alpha(
            model.gdh_write[i],
            x,
            sidecar,
            n_write_heads=gdh_heads,
            route_topk=route_topk,
            eps=1e-6,
        )

        # Routing usage balancing (layer-local): encourage uniform slot usage.
        # Fix: Use SOFT probs (gradient flow to dead slots) AND gate-aware weighting (ignore garbage).
        gate_weight = g.detach() if g is not None else torch.ones_like(alpha_soft[..., :1, :1])
        weighted_sum = (alpha_soft * gate_weight).sum(dim=(0, 1, 2))  # [R]
        total_weight = gate_weight.sum() * gdh_heads
        
        usage = weighted_sum / (total_weight + 1e-9)
        usage_target = torch.full_like(usage, 1.0 / usage.shape[0])
        usage_losses.append((usage - usage_target).pow(2).mean())

        sidecar = sidecar + _scan_accumulate(delta, scan_beta=scan_beta)

        if collect_sidecar:
            layer_slot_cos.append(_pairwise_slot_cos_last_mean(sidecar))

    x = gpt_norm(x)
    logits = model.lm_head(x)[..., :model.config.vocab_size].float()
    softcap = 15
    logits = softcap * torch.tanh(logits / softcap)

    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    usage_loss = torch.stack(usage_losses).mean() if usage_losses else ce_loss.new_zeros(())
    total_loss = ce_loss + usage_balance_lambda * usage_loss

    metrics = _compute_answer_metrics(logits, targets, eval_topk=eval_topk)
    stats = {"slot_cos_mean_last_per_layer": layer_slot_cos} if collect_sidecar else None
    return total_loss, ce_loss, usage_loss, metrics, stats


def run_one_beta(args: argparse.Namespace, beta: float, device: str) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = _build_model(args, device=device)
    model.train()

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
        f"[beta={beta}] lr_schedule base={base_lr:.6g} min={min_lr:.6g} decay_iters={lr_decay_iters}",
        flush=True,
    )

    use_cuda = device == "cuda"
    ac = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

    # Fixed eval batch for stable traces.
    torch.manual_seed(args.seed + 1)
    eval_idx, eval_tgt = make_mqar_batch(args, batch_size=args.eval_batch_size, device=device)

    history: list[dict[str, Any]] = []

    def eval_record(
        step: int,
        train_total: float | None = None,
        train_ce: float | None = None,
        train_usage: float | None = None,
        train_acc1: float | None = None,
        train_mrr: float | None = None,
    ):
        model.eval()
        with torch.no_grad():
            with ac:
                if args.arch == "baseline":
                    eval_total, eval_ce, eval_usage, eval_metrics, stats = _forward_baseline(
                        model,
                        eval_idx,
                        eval_tgt,
                        eval_topk=args.eval_topk,
                    )
                else:
                    eval_total, eval_ce, eval_usage, eval_metrics, stats = _forward_gdh(
                        model,
                        eval_idx,
                        eval_tgt,
                        scan_beta=beta,
                        route_topk=args.route_topk,
                        usage_balance_lambda=args.usage_balance_lambda,
                        eval_topk=args.eval_topk,
                        collect_sidecar=True,
                    )

        rec: dict[str, Any] = {
            "step": step,
            "beta": beta,
            "lr": float(opt.param_groups[0]["lr"]),
            "eval_total": float(eval_total.item()),
            "eval_ce": float(eval_ce.item()),
            "eval_usage_loss": float(eval_usage.item()),
            "eval_acc_top1": float(eval_metrics["acc_top1"]),
            "eval_mrr": float(eval_metrics["mrr"]),
            "eval_n_answers": float(eval_metrics["n_answers"]),
            # compatibility alias
            "eval_acc": float(eval_metrics["acc_top1"]),
        }
        topk_key = f"acc_top{max(1, min(args.eval_topk, args.vocab_size))}"
        rec[f"eval_{topk_key}"] = float(eval_metrics[topk_key])

        if args.arch == "gdh":
            rec["slot_cos_l_last"] = float(stats["slot_cos_mean_last_per_layer"][-1])
            rec["slot_cos_layers"] = [float(v) for v in stats["slot_cos_mean_last_per_layer"]]
        else:
            rec["slot_cos_l_last"] = None
            rec["slot_cos_layers"] = []

        if train_total is not None:
            rec["train_total"] = float(train_total)
        if train_ce is not None:
            rec["train_ce"] = float(train_ce)
        if train_usage is not None:
            rec["train_usage_loss"] = float(train_usage)
        if train_acc1 is not None:
            rec["train_acc_top1"] = float(train_acc1)
        if train_mrr is not None:
            rec["train_mrr"] = float(train_mrr)

        history.append(rec)

        # Progress line (helps keep long runs alive and visible).
        cos_txt = "n/a" if rec["slot_cos_l_last"] is None else f"{rec['slot_cos_l_last']:.4f}"
        print(
            f"[beta={beta}] step={step} eval_acc1={rec['eval_acc_top1']:.4f} "
            f"eval_mrr={rec['eval_mrr']:.4f} eval_ce={rec['eval_ce']:.4f} "
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
                total, ce, usage, train_metrics, _ = _forward_baseline(
                    model,
                    idx,
                    tgt,
                    eval_topk=args.eval_topk,
                )
            else:
                total, ce, usage, train_metrics, _ = _forward_gdh(
                    model,
                    idx,
                    tgt,
                    scan_beta=beta,
                    route_topk=args.route_topk,
                    usage_balance_lambda=args.usage_balance_lambda,
                    eval_topk=args.eval_topk,
                    collect_sidecar=False,
                )
        total.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps:
            eval_record(
                step=step,
                train_total=float(total.item()),
                train_ce=float(ce.item()),
                train_usage=float(usage.item()),
                train_acc1=float(train_metrics["acc_top1"]),
                train_mrr=float(train_metrics["mrr"]),
            )

    first = history[0]
    last = history[-1]
    return {
        "beta": beta,
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

        topk_label = f"eval_acc_top{max(1, min(payload['config']['eval_topk'], payload['config']['vocab_size']))}"

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

        fig.suptitle(f"MQAR probe ({payload['config']['arch']})", y=0.995)
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
    p = argparse.ArgumentParser(description="MQAR scan-beta probe (baseline/GDH)")

    # Sweep config.
    p.add_argument("--arch", type=str, default="gdh", choices=["baseline", "gdh"])
    p.add_argument("--betas", type=str, default="1.0,0.99,0.95,0.9")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)

    # Model config.
    p.add_argument("--sequence-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-embd", type=int, default=128)

    # GDH-only config.
    p.add_argument("--gdh-slots", type=int, default=8)
    p.add_argument("--gdh-write-heads", type=int, default=8)
    p.add_argument("--gdh-use-read-gate", action="store_true", help="enable GDH read gate")
    p.add_argument("--route-topk", type=int, default=0)
    p.add_argument("--usage-balance-lambda", type=float, default=0.0)

    # SWA config (Stage-1 blindfold).
    p.add_argument("--swa-window", type=int, default=0, help="Sliding window size (0=off). If set, forces SWA on all layers.")

    # Data config.
    p.add_argument("--n-pairs", type=int, default=16)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--gap-min", type=int, default=64)
    p.add_argument("--gap-max", type=int, default=192)

    # Vocab partitions.
    p.add_argument("--key-vocab", type=int, default=2048)
    p.add_argument("--value-vocab", type=int, default=2048)
    p.add_argument("--query-offset", type=int, default=4096)
    p.add_argument("--filler-offset", type=int, default=6144)
    p.add_argument("--filler-vocab", type=int, default=2048)

    # Train/eval.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-topk", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-decay-iters", type=int, default=-1, help="cosine LR decay horizon in optimizer steps (-1 => use --steps)")
    p.add_argument("--min-lr", type=float, default=0.0, help="minimum LR at/after decay horizon")

    # Optional strict stress requirement.
    p.add_argument("--enforce-capacity-stress", action="store_true", help="require n_pairs > gdh_slots")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Safety checks.
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
        # beta is not used in baseline path; keep one for consistent output.
        betas = [betas[0]]

    runs = []
    for beta in betas:
        runs.append(run_one_beta(args, beta=beta, device=args.device))

    topk_key = f"acc_top{max(1, min(args.eval_topk, args.vocab_size))}"

    summary_rows = []
    for r in runs:
        summary_rows.append({
            "beta": r["beta"],
            "eval_acc_top1_0": r["first"]["eval_acc_top1"],
            "eval_acc_top1_last": r["last"]["eval_acc_top1"],
            f"eval_{topk_key}_last": r["last"][f"eval_{topk_key}"],
            "eval_mrr_last": r["last"]["eval_mrr"],
            "eval_ce_last": r["last"]["eval_ce"],
            "slot_cos_last": r["last"]["slot_cos_l_last"],
        })

    payload = {
        "config": vars(args),
        "betas": betas,
        "summary": summary_rows,
        "runs": runs,
    }

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"mqar_scan_beta_probe_{ts}.json")
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
