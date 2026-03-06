"""
GDH (Gated Double Helix) core module.

Current scope:
- Dense v1 layer forward (oracle-aligned)
- Input/shape contract validation

Notes:
- Implementation is intentionally conservative.
- Process uses a simple causal toy path in this phase.
- Decomposed translators and faster kernels come later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GDHConfig:
    """Configuration for one GDH layer.

    - n_embd: model width D
    - n_slots: sidecar slot count R
    - n_write_heads: write head count h
    - use_read_gate: if False, disable read gating (always-on sidecar read path)
    - use_write_brain: if True, apply write-space MLP residual (Linear->ReLU²->Linear)
    - write_brain_hidden_mult: hidden width multiplier for write brain (hidden = mult * D)
    - lora_rank: reserved for upcoming decomposed path
    - eps: RMSNorm epsilon
    """

    n_embd: int
    n_slots: int
    n_write_heads: int
    use_read_gate: bool = True
    use_write_brain: bool = False
    write_brain_hidden_mult: int = 4
    lora_rank: int = 8
    eps: float = 1e-6

    def validate(self) -> None:
        if self.n_embd <= 0:
            raise ValueError("n_embd must be > 0")
        if self.n_slots <= 0:
            raise ValueError("n_slots must be > 0")
        if self.n_write_heads <= 0:
            raise ValueError("n_write_heads must be > 0")
        if self.n_embd % self.n_write_heads != 0:
            raise ValueError("n_embd must be divisible by n_write_heads")
        if self.write_brain_hidden_mult <= 0:
            raise ValueError("write_brain_hidden_mult must be > 0")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be > 0")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class GDHTensorContract:
    """Reference tensor contract for one GDH forward."""

    local: str = "[B, N, D]"
    sidecar_prev: str = "[B, N, R, D]"
    boundary_mask: str = "[B, N] optional bool"
    local_out: str = "[B, N, D]"
    delta: str = "[B, N, R, D]"
    sidecar_curr: str = "[B, N, R, D]"


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _cosine_logit(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cosine-style projection logit with controlled scale.

    x: [..., D], w: [D, 1] -> output [...]
    """
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    w_col = w.squeeze(-1)
    w_unit = w_col / w_col.norm().clamp_min(eps)
    return torch.einsum("...d,d->...", x_unit, w_unit)


def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cosine similarity for two tensors with shared trailing dim D.

    x, y: [..., D] -> output [...]
    """
    x_unit = x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    y_unit = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (x_unit * y_unit).sum(dim=-1)


def validate_gdh_inputs(
    local: torch.Tensor,
    sidecar_prev: torch.Tensor,
    config: GDHConfig,
    boundary_mask: Optional[torch.Tensor] = None,
) -> None:
    """Validate shape contract for GDH forward."""
    config.validate()

    if local.ndim != 3:
        raise ValueError(f"local must be [B,N,D], got shape {tuple(local.shape)}")
    if sidecar_prev.ndim != 4:
        raise ValueError(f"sidecar_prev must be [B,N,R,D], got shape {tuple(sidecar_prev.shape)}")

    b, n, d = local.shape
    b2, n2, r, d2 = sidecar_prev.shape

    if (b, n, d) != (b2, n2, d2):
        raise ValueError(
            "local and sidecar_prev dimensions mismatch: "
            f"local={tuple(local.shape)}, sidecar_prev={tuple(sidecar_prev.shape)}"
        )

    if d != config.n_embd:
        raise ValueError(f"local D={d} does not match config.n_embd={config.n_embd}")
    if r != config.n_slots:
        raise ValueError(f"sidecar R={r} does not match config.n_slots={config.n_slots}")

    if boundary_mask is not None and boundary_mask.shape != (b, n):
        raise ValueError(f"boundary_mask must be [B,N]={b,n}, got {tuple(boundary_mask.shape)}")


class GDHReadCore(nn.Module):
    """Read phase core: sidecar -> local."""

    def __init__(self, d: int, *, use_read_gate: bool = True, use_read_mute_gate: bool = True):
        super().__init__()
        # Legacy arg retained for backward compatibility; competition read gate removed.
        self.use_read_mute_gate = use_read_mute_gate
        self.W_q_read = nn.Parameter(torch.empty(d, d))
        self.W_k_read_global = nn.Parameter(torch.empty(d, d))
        self.W_v_read_global = nn.Parameter(torch.empty(d, d))
        self.W_o_read = nn.Parameter(torch.empty(d, d))
        # Scalar mute gate for sidecar read injection (per token).
        self.W_g_read_mute = nn.Parameter(torch.empty(d, 1))
        self.b_g_read_mute = nn.Parameter(torch.zeros(1))

    def reset_parameters(self, *, std: float, zero_init_mixer: bool) -> None:
        s = math.sqrt(3.0) * std
        nn.init.uniform_(self.W_q_read, -s, s)
        nn.init.uniform_(self.W_k_read_global, -s, s)
        nn.init.uniform_(self.W_v_read_global, -s, s)
        nn.init.zeros_(self.W_g_read_mute)
        nn.init.constant_(self.b_g_read_mute, -1.0)
        # v2.3 ReZero output coupling: sidecar read path starts silent on residual stream.
        nn.init.zeros_(self.W_o_read)

    def forward_step(self, l_t: torch.Tensor, s_t_prev: torch.Tensor, *, eps: float) -> torch.Tensor:
        d = l_t.shape[-1]
        x_read = _rms_norm(l_t, eps=eps)
        q_loc = x_read @ self.W_q_read

        s_hat = _rms_norm(s_t_prev, eps=eps)
        k_mem = s_hat @ self.W_k_read_global
        v_mem = s_hat @ self.W_v_read_global

        logits_read = (k_mem @ q_loc) / math.sqrt(d)
        alpha_read = torch.softmax(logits_read, dim=0)
        z_read = alpha_read @ v_mem

        z_proj = z_read @ self.W_o_read
        if self.use_read_mute_gate:
            read_mute = 0.05 + 0.95 * torch.sigmoid((x_read @ self.W_g_read_mute + self.b_g_read_mute).squeeze(-1))
            z_proj = read_mute * z_proj

        return l_t + z_proj

    def forward_sequence(self, local: torch.Tensor, sidecar_prev: torch.Tensor, *, eps: float) -> torch.Tensor:
        """Vectorized read over [B,N,D] and [B,N,R,D]."""
        bsz, n_tokens, d = local.shape
        assert sidecar_prev.shape[:2] == (bsz, n_tokens)

        x_read = _rms_norm(local, eps=eps)                                        # [B,N,D]
        q_loc = torch.matmul(x_read, self.W_q_read)                               # [B,N,D]

        s_hat = _rms_norm(sidecar_prev, eps=eps)                                  # [B,N,R,D]
        k_mem = torch.einsum("bnrd,df->bnrf", s_hat, self.W_k_read_global)       # [B,N,R,D]
        v_mem = torch.einsum("bnrd,df->bnrf", s_hat, self.W_v_read_global)       # [B,N,R,D]

        logits = torch.einsum("bnd,bnrd->bnr", q_loc, k_mem) / math.sqrt(d)      # [B,N,R]
        alpha = torch.softmax(logits, dim=-1)                                     # [B,N,R]
        z_read = torch.einsum("bnr,bnrd->bnd", alpha, v_mem)                     # [B,N,D]

        z_proj = torch.matmul(z_read, self.W_o_read)                              # [B,N,D]
        if self.use_read_mute_gate:
            read_mute = 0.05 + 0.95 * torch.sigmoid(torch.matmul(x_read, self.W_g_read_mute) + self.b_g_read_mute)   # [B,N,1]
            z_proj = read_mute * z_proj

        return local + z_proj                                                        # [B,N,D]


class GDHProcessCore(nn.Module):
    """Process phase core: causal local refinement."""

    def __init__(self, d: int):
        super().__init__()
        self.W_self_q = nn.Parameter(torch.empty(d, d))
        self.W_self_k = nn.Parameter(torch.empty(d, d))
        self.W_self_v = nn.Parameter(torch.empty(d, d))
        self.W_self_o = nn.Parameter(torch.empty(d, d))
        self.W_mlp_in = nn.Parameter(torch.empty(d, d))
        self.W_mlp_out = nn.Parameter(torch.empty(d, d))

    def reset_parameters(self, *, std: float) -> None:
        nn.init.normal_(self.W_self_q, mean=0.0, std=std)
        nn.init.normal_(self.W_self_k, mean=0.0, std=std)
        nn.init.normal_(self.W_self_v, mean=0.0, std=std)
        nn.init.normal_(self.W_self_o, mean=0.0, std=std)
        nn.init.normal_(self.W_mlp_in, mean=0.0, std=std)
        nn.init.normal_(self.W_mlp_out, mean=0.0, std=std)

    def forward_step(self, l_tilde_hist: list[torch.Tensor], *, eps: float) -> torch.Tensor:
        hist = torch.stack(l_tilde_hist, dim=0)  # [t+1, D]
        hist_norm = _rms_norm(hist, eps=eps)
        d = hist_norm.shape[-1]

        q_self = hist_norm[-1] @ self.W_self_q
        k_self = hist_norm @ self.W_self_k
        v_self = hist_norm @ self.W_self_v

        logits_self = (k_self @ q_self) / math.sqrt(d)
        alpha_self = torch.softmax(logits_self, dim=0)
        ctx = alpha_self @ v_self

        l_tilde_t = l_tilde_hist[-1]
        l_hat = l_tilde_t + (ctx @ self.W_self_o)
        ff = torch.relu(_rms_norm(l_hat, eps=eps) @ self.W_mlp_in).square()
        return l_hat + (ff @ self.W_mlp_out)


class GDHWriteCore(nn.Module):
    """Write phase core using learnable slot addresses (E_slots).

    Routing design:
    - static slot queries (global):  Q_slots = RMSNorm(E_slots) @ W_q_slots_global
    - token-conditioned keys/values (layer-local):
        K_upd = x_write @ W_k_write
        V_upd = x_write @ W_v_write
    - slot routing weights come from softmax(Q_slots · K_upd)

    Optional write-brain:
    - Applies sidecar-space residual MLP on write deltas:
      delta = delta + MLP(RMSNorm(delta)), where MLP is Linear->ReLU²->Linear.
    """

    def __init__(
        self,
        d: int,
        r: int | None = None,
        *,
        use_write_brain: bool = False,
        write_brain_hidden_mult: int = 4,
    ):
        super().__init__()
        if r is None or r <= 0:
            raise ValueError("GDHWriteCore requires positive slot count r")

        # Global slot-address parameters (tied across layers at GPT level).
        self.E_slots = nn.Parameter(torch.empty(r, d))
        self.W_q_slots_global = nn.Parameter(torch.empty(d, d))

        # Layer-local token translators + output mixer.
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

    @property
    def W_q_write_global(self) -> nn.Parameter:
        """Legacy alias for backward compatibility; use W_q_slots_global."""
        return self.W_q_slots_global

    @W_q_write_global.setter
    def W_q_write_global(self, value: nn.Parameter) -> None:
        self.W_q_slots_global = value

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backward-compat for checkpoints saved before naming cleanup.
        legacy_key = prefix + "W_q_write_global"
        new_key = prefix + "W_q_slots_global"
        if legacy_key in state_dict and new_key not in state_dict:
            state_dict[new_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def reset_parameters(self, *, std: float, zero_init_mixer: bool) -> None:
        # v2.3: match mainline matrix init family (uniform with equivalent std).
        s = math.sqrt(3.0) * std
        nn.init.uniform_(self.E_slots, -s, s)
        nn.init.uniform_(self.W_q_slots_global, -s, s)

        nn.init.uniform_(self.W_k_write, -s, s)
        nn.init.uniform_(self.W_v_write, -s, s)
        # Keep write output projection active (do NOT zero) so sidecar state is alive from step 1.
        nn.init.uniform_(self.W_o_write, -s, s)

        if self.use_write_brain and self.W_write_mlp_in_global is not None:
            nn.init.uniform_(self.W_write_mlp_in_global, -s, s)
            # ReZero output coupling for write-brain residual branch.
            nn.init.zeros_(self.W_write_mlp_out_global)

    def _apply_write_brain(self, delta: torch.Tensor, *, eps: float) -> torch.Tensor:
        if not self.use_write_brain or self.W_write_mlp_in_global is None:
            return delta
        delta_norm = _rms_norm(delta, eps=eps)
        hidden = torch.relu(torch.matmul(delta_norm, self.W_write_mlp_in_global)).square()
        return delta + torch.matmul(hidden, self.W_write_mlp_out_global)

    def forward_step(
        self,
        l_out_t: torch.Tensor,
        s_t_prev: torch.Tensor,
        *,
        n_write_heads: int,
        eps: float,
    ) -> torch.Tensor:
        d = l_out_t.shape[-1]
        r = s_t_prev.shape[0]
        assert self.E_slots.shape[0] == r, "sidecar slot count must match E_slots"
        d_h = d // n_write_heads

        x_write = _rms_norm(l_out_t, eps=eps)
        k_upd = x_write @ self.W_k_write
        v_upd = x_write @ self.W_v_write

        e_slots = _rms_norm(self.E_slots, eps=eps)
        q_slots = e_slots @ self.W_q_slots_global

        q_h = q_slots.view(r, n_write_heads, d_h)   # [R,h,d_h]
        k_h = k_upd.view(n_write_heads, d_h)        # [h,d_h]
        v_h = v_upd.view(n_write_heads, d_h)        # [h,d_h]

        delta_raw = torch.zeros(r, d, dtype=l_out_t.dtype, device=l_out_t.device)
        for j in range(n_write_heads):
            q_slots_j = q_h[:, j, :]                                # [R,d_h]
            logits_w = (q_slots_j @ k_h[j]) / math.sqrt(d_h)        # [R]
            alpha_w = torch.softmax(logits_w, dim=0)                # [R]
            delta_raw[:, j * d_h:(j + 1) * d_h] = alpha_w[:, None] * v_h[j][None, :]

        delta = delta_raw @ self.W_o_write
        return self._apply_write_brain(delta, eps=eps)

    def forward_sequence_delta(
        self,
        local_out: torch.Tensor,
        sidecar_prev: torch.Tensor,
        *,
        n_write_heads: int,
        eps: float,
    ) -> torch.Tensor:
        """Vectorized write delta for [B,N,D] + [B,N,R,D] -> [B,N,R,D]."""
        bsz, n_tokens, d = local_out.shape
        assert d % n_write_heads == 0
        assert sidecar_prev.shape[:2] == (bsz, n_tokens)
        r = sidecar_prev.shape[2]
        assert self.E_slots.shape[0] == r, "sidecar slot count must match E_slots"
        d_h = d // n_write_heads

        x_write = _rms_norm(local_out, eps=eps)                                 # [B,N,D]
        k_upd = torch.matmul(x_write, self.W_k_write)                           # [B,N,D]
        v_upd = torch.matmul(x_write, self.W_v_write)                           # [B,N,D]

        e_slots = _rms_norm(self.E_slots, eps=eps)                              # [R,D]
        q_slots = torch.matmul(e_slots, self.W_q_slots_global)                   # [R,D]
        q_slots_h = q_slots.view(r, n_write_heads, d_h)                         # [R,h,d_h]

        k_h = k_upd.view(bsz, n_tokens, n_write_heads, d_h)                     # [B,N,h,d_h]
        v_h = v_upd.view(bsz, n_tokens, n_write_heads, d_h)                     # [B,N,h,d_h]

        logits = torch.einsum("rhd,bnhd->bnhr", q_slots_h, k_h) / math.sqrt(d_h)   # [B,N,h,R]
        alpha = torch.softmax(logits, dim=-1)                                         # [B,N,h,R]
        delta_heads = torch.einsum("bnhr,bnhd->bnhrd", alpha, v_h)                   # [B,N,h,R,d_h]

        delta_raw = delta_heads.permute(0, 1, 3, 2, 4).contiguous().view(bsz, n_tokens, r, d)
        delta = torch.einsum("bnrd,df->bnrf", delta_raw, self.W_o_write)
        return self._apply_write_brain(delta, eps=eps)


class GDHLayer(nn.Module):
    """Dense GDH layer (v1, oracle-aligned).

    Forward inputs:
    - local: [B, N, D]
    - sidecar_prev: [B, N, R, D]
    - boundary_mask: optional [B, N] bool

    Forward outputs:
    - local_out: [B, N, D]
    - delta: [B, N, R, D]
    - sidecar_curr: [B, N, R, D]
    """

    def __init__(self, config: GDHConfig, *, zero_init_mixers: bool = False):
        super().__init__()
        config.validate()
        self.config = config

        d = config.n_embd
        r = config.n_slots

        self.read = GDHReadCore(d, use_read_gate=config.use_read_gate)
        self.process = GDHProcessCore(d)
        self.write = GDHWriteCore(
            d,
            r,
            use_write_brain=config.use_write_brain,
            write_brain_hidden_mult=config.write_brain_hidden_mult,
        )

        self.reset_parameters(zero_init_mixers=zero_init_mixers)

    def reset_parameters(self, *, zero_init_mixers: bool = False) -> None:
        std = self.config.n_embd ** -0.5
        self.read.reset_parameters(std=std, zero_init_mixer=zero_init_mixers)
        self.process.reset_parameters(std=std)
        self.write.reset_parameters(std=std, zero_init_mixer=zero_init_mixers)

    def forward(
        self,
        local: torch.Tensor,
        sidecar_prev: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        validate_gdh_inputs(local, sidecar_prev, self.config, boundary_mask)

        bsz, n_tokens, d = local.shape
        _, _, n_slots, _ = sidecar_prev.shape

        if boundary_mask is None:
            boundary_mask = torch.zeros(bsz, n_tokens, dtype=torch.bool, device=local.device)
        else:
            boundary_mask = boundary_mask.to(torch.bool)

        h = self.config.n_write_heads
        eps = self.config.eps

        local_out = torch.empty_like(local)
        delta_all = torch.zeros(bsz, n_tokens, n_slots, d, dtype=local.dtype, device=local.device)
        sidecar_curr = torch.empty_like(sidecar_prev)

        for b in range(bsz):
            running = torch.zeros(n_slots, d, dtype=local.dtype, device=local.device)
            l_tilde_hist: list[torch.Tensor] = []

            for t in range(n_tokens):
                l_t = local[b, t]
                s_t_prev = sidecar_prev[b, t]

                l_tilde = self.read.forward_step(l_t, s_t_prev, eps=eps)

                l_tilde_hist.append(l_tilde)
                l_out_t = self.process.forward_step(l_tilde_hist, eps=eps)
                local_out[b, t] = l_out_t

                delta_t = self.write.forward_step(
                    l_out_t,
                    s_t_prev,
                    n_write_heads=h,
                    eps=eps,
                )
                delta_all[b, t] = delta_t

                if boundary_mask[b, t].item():
                    running.zero_()
                running = running + delta_t
                sidecar_curr[b, t] = s_t_prev + running

        return local_out, delta_all, sidecar_curr
