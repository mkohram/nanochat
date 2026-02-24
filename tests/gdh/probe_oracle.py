"""Slow oracle for MQAR GDH probe behavior.

This oracle mirrors the *probe path* in `experiments/mqar_scan_beta_probe.py`
(including sparse top-k routing, gate-aware usage balancing, ad-hoc g_write,
and leaky scan), but with explicit/slow computations where practical.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from nanochat.gpt import norm as gpt_norm


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def scan_accumulate_oracle(delta: torch.Tensor, scan_beta: float) -> torch.Tensor:
    """Slow/reference token-wise accumulation used in MQAR probe.

    - beta >= 1.0: pure cumsum
    - beta <= 0.0: passthrough delta
    - otherwise: y_t = beta*y_{t-1} + delta_t
    """
    if scan_beta >= 1.0:
        return torch.cumsum(delta, dim=1)
    if scan_beta <= 0.0:
        return delta

    bsz, n_tokens, r, d = delta.shape
    out = torch.zeros_like(delta)
    running = torch.zeros(bsz, r, d, device=delta.device, dtype=delta.dtype)
    beta = torch.tensor(float(scan_beta), device=delta.device, dtype=delta.dtype)

    for t in range(n_tokens):
        running = beta * running + delta[:, t]
        out[:, t] = running
    return out


def write_delta_and_alpha_oracle(
    write_core,
    local_out: torch.Tensor,
    *,
    n_write_heads: int,
    route_topk: int,
    eps: float,
    gate_proj: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reference write-delta path matching probe behavior.

    Returns:
      delta: [B,N,R,D]
      alpha: [B,N,h,R]       (hard/masked, possibly renormalized)
      alpha_soft: [B,N,h,R]  (dense softmax)
      g_write: [B,N,1,1] or None
    """
    bsz, n_tokens, d = local_out.shape
    r = write_core.E_slots.shape[0]
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
    if gate_proj is not None:
        g_write = torch.sigmoid(gate_proj(x_write)).unsqueeze(-2)  # [B,N,1,1]
        delta = delta * g_write

    return delta, alpha, alpha_soft, g_write


def usage_balance_loss_oracle(
    alpha_soft: torch.Tensor,
    g_write: torch.Tensor | None,
    *,
    n_write_heads: int,
) -> torch.Tensor:
    """Layer-local usage balancing loss used in probe path."""
    gate_weight = g_write.detach() if g_write is not None else torch.ones_like(alpha_soft[..., :1, :1])
    weighted_sum = (alpha_soft * gate_weight).sum(dim=(0, 1, 2))
    total_weight = gate_weight.sum() * n_write_heads

    usage = weighted_sum / (total_weight + 1e-9)
    usage_target = torch.full_like(usage, 1.0 / usage.shape[0])
    return (usage - usage_target).pow(2).mean()


def _pairwise_slot_cos_last_mean(sidecar: torch.Tensor, eps: float = 1e-9) -> float:
    s = sidecar.float()[:, -1]  # [B,R,D]
    s = s / s.norm(dim=-1, keepdim=True).clamp_min(eps)
    gram = torch.einsum("brd,bsd->brs", s, s)
    r = gram.shape[-1]
    mask = ~torch.eye(r, dtype=torch.bool, device=gram.device)
    vals = gram[:, mask]
    return float(vals.mean().item())


def _compute_answer_metrics(logits: torch.Tensor, targets: torch.Tensor, eval_topk: int) -> dict[str, float]:
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

        sel_logits = logits[valid]
        sel_targets = targets[valid].long()

        pred1 = sel_logits.argmax(dim=-1)
        acc1 = float((pred1 == sel_targets).float().mean().item())

        k = max(1, min(eval_topk, sel_logits.shape[-1]))
        topk = torch.topk(sel_logits, k=k, dim=-1).indices
        hitk = (topk == sel_targets.unsqueeze(1)).any(dim=1)
        acck = float(hitk.float().mean().item())

        target_scores = sel_logits.gather(1, sel_targets.unsqueeze(1))
        ranks = (sel_logits > target_scores).sum(dim=1) + 1
        mrr = float((1.0 / ranks.float()).mean().item())

        return {
            "acc_top1": acc1,
            f"acc_top{k}": acck,
            "mrr": mrr,
            "n_answers": float(n_valid),
        }


def forward_gdh_probe_oracle(
    model,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    scan_beta: float,
    route_topk: int,
    usage_balance_lambda: float,
    eval_topk: int,
    collect_sidecar: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict[str, Any] | None]:
    """Slow oracle for probe `_forward_gdh` behavior.

    Note: This intentionally mirrors probe logic, but evaluates read in explicit loops
    via `forward_step` to provide an independent path from `forward_sequence`.
    """
    bsz, n_tokens = idx.size()

    x = model.transformer.wte(idx)
    x = gpt_norm(x)
    x0 = x

    gdh_slots = model.config.gdh_slots
    gdh_heads = model.config.n_head if model.config.gdh_write_heads <= 0 else model.config.gdh_write_heads

    sidecar = torch.zeros(bsz, n_tokens, gdh_slots, model.config.n_embd, device=x.device, dtype=x.dtype)
    cos_sin = model.cos[:, :n_tokens], model.sin[:, :n_tokens]

    layer_slot_cos: list[float] = []
    usage_losses: list[torch.Tensor] = []

    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0

        # Explicit read loop for oracle independence.
        x_read = torch.empty_like(x)
        for b in range(bsz):
            for t in range(n_tokens):
                x_read[b, t] = model.gdh_read[i].forward_step(x[b, t], sidecar[b, t], eps=1e-6)
        x = x_read

        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i], kv_cache=None)

        gate_proj = model.g_write_projs[i] if hasattr(model, "g_write_projs") else None
        delta, _alpha, alpha_soft, g = write_delta_and_alpha_oracle(
            model.gdh_write[i],
            x,
            n_write_heads=gdh_heads,
            route_topk=route_topk,
            eps=1e-6,
            gate_proj=gate_proj,
        )

        usage_losses.append(usage_balance_loss_oracle(alpha_soft, g, n_write_heads=gdh_heads))

        sidecar = sidecar + scan_accumulate_oracle(delta, scan_beta=scan_beta)

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
