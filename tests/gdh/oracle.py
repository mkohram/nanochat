"""
Slow reference oracle for a single GDH layer.

This module is intentionally explicit and loop-based. It is not optimized.
It exists to provide a correctness target for later parallelized implementations.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class GDHOracleParams:
    # Read
    W_q_read: torch.Tensor            # [D, D]
    W_k_read_global: torch.Tensor     # [D, D]
    W_v_read_global: torch.Tensor     # [D, D]
    W_o_read: torch.Tensor            # [D, D]
    W_g_read_mute: torch.Tensor       # [D, 1]
    b_g_read_mute: torch.Tensor       # [1]

    # Process (causal toy transformer step)
    W_self_q: torch.Tensor            # [D, D]
    W_self_k: torch.Tensor            # [D, D]
    W_self_v: torch.Tensor            # [D, D]
    W_self_o: torch.Tensor            # [D, D]
    W_mlp_in: torch.Tensor            # [D, D]
    W_mlp_out: torch.Tensor           # [D, D]

    # Write (slot-address routing)
    E_slots: torch.Tensor             # [R, D] permanent learnable slot addresses
    W_q_slots_global: torch.Tensor    # [D, D] projects slot addresses into write queries
    W_k_write: torch.Tensor           # [D, D] projects local token output into write keys
    W_v_write: torch.Tensor           # [D, D] projects local token output into write values
    W_o_write: torch.Tensor           # [D, D]


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def make_oracle_params(
    d: int,
    n_slots: int,
    r: int,
    h: int,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    zero_init_mixers: bool = False,
) -> GDHOracleParams:
    """Create deterministic tiny parameters for oracle tests."""
    del r

    torch.manual_seed(seed)

    def rand(*shape, scale=0.02):
        return torch.randn(*shape, device=device, dtype=dtype) * scale

    W_o_read = torch.zeros(d, d, device=device, dtype=dtype) if zero_init_mixers else rand(d, d)
    W_o_write = torch.zeros(d, d, device=device, dtype=dtype) if zero_init_mixers else rand(d, d)

    return GDHOracleParams(
        W_q_read=rand(d, d),
        W_k_read_global=rand(d, d),
        W_v_read_global=rand(d, d),
        W_o_read=W_o_read,
        W_g_read_mute=rand(d, 1),
        b_g_read_mute=torch.zeros(1, device=device, dtype=dtype),
        W_self_q=rand(d, d),
        W_self_k=rand(d, d),
        W_self_v=rand(d, d),
        W_self_o=rand(d, d),
        W_mlp_in=rand(d, d),
        W_mlp_out=rand(d, d),
        E_slots=rand(n_slots, d),
        W_q_slots_global=rand(d, d),
        W_k_write=rand(d, d),
        W_v_write=rand(d, d),
        W_o_write=W_o_write,
    )


def gdh_oracle_layer(
    L_in: torch.Tensor,
    S_prev: torch.Tensor,
    params: GDHOracleParams,
    *,
    n_write_heads: int,
    boundary_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slow reference forward for one GDH layer.

    Args:
        L_in: [B, N, D]
        S_prev: [B, N, R, D]
        params: oracle parameters
        n_write_heads: h (must divide D)
        boundary_mask: [B, N] bool/int, where 1 means segment start/reset running cumsum

    Returns:
        L_out: [B, N, D]
        delta_all: [B, N, R, D]
        S_curr: [B, N, R, D]
    """
    assert L_in.ndim == 3, f"L_in must be [B,N,D], got {tuple(L_in.shape)}"
    assert S_prev.ndim == 4, f"S_prev must be [B,N,R,D], got {tuple(S_prev.shape)}"

    B, N, D = L_in.shape
    B2, N2, R, D2 = S_prev.shape
    assert (B, N, D) == (B2, N2, D2), "L_in and S_prev shape mismatch"
    assert D % n_write_heads == 0, "D must be divisible by n_write_heads"

    if boundary_mask is None:
        boundary_mask = torch.zeros(B, N, dtype=torch.bool, device=L_in.device)
    else:
        assert boundary_mask.shape == (B, N), f"boundary_mask must be [B,N], got {tuple(boundary_mask.shape)}"
        boundary_mask = boundary_mask.to(torch.bool)

    d_h = D // n_write_heads

    L_out = torch.empty_like(L_in)
    delta_all = torch.zeros(B, N, R, D, dtype=L_in.dtype, device=L_in.device)
    S_curr = torch.empty_like(S_prev)

    for b in range(B):
        running = torch.zeros(R, D, dtype=L_in.dtype, device=L_in.device)
        l_tilde_hist: list[torch.Tensor] = []

        for t in range(N):
            l_t = L_in[b, t]          # [D]
            s_t_prev = S_prev[b, t]   # [R, D]

            # ---------------------
            # Phase I: Read
            # ---------------------
            x_read = _rms_norm(l_t, eps=eps)                      # [D]
            q_loc = x_read @ params.W_q_read                      # [D]

            s_hat = _rms_norm(s_t_prev, eps=eps)                  # [R, D]
            k_mem = s_hat @ params.W_k_read_global                # [R, D]
            v_mem = s_hat @ params.W_v_read_global                # [R, D]

            logits_read = (k_mem @ q_loc) / math.sqrt(D)          # [R]
            alpha_read = torch.softmax(logits_read, dim=0)        # [R]
            z_read = alpha_read @ v_mem                           # [D]

            g_read_mute = 0.05 + 0.95 * torch.sigmoid((x_read @ params.W_g_read_mute + params.b_g_read_mute).squeeze(-1))  # scalar
            l_tilde = l_t + g_read_mute * (z_read @ params.W_o_read)   # [D]

            # ---------------------
            # Phase II: Process (causal)
            # ---------------------
            l_tilde_hist.append(l_tilde)
            h = torch.stack(l_tilde_hist, dim=0)                  # [t+1, D]
            h_norm = _rms_norm(h, eps=eps)                        # [t+1, D]

            q_self = h_norm[-1] @ params.W_self_q                 # [D]
            k_self = h_norm @ params.W_self_k                     # [t+1, D]
            v_self = h_norm @ params.W_self_v                     # [t+1, D]

            logits_self = (k_self @ q_self) / math.sqrt(D)        # [t+1]
            alpha_self = torch.softmax(logits_self, dim=0)        # [t+1]
            ctx = alpha_self @ v_self                             # [D]

            l_hat = l_tilde + (ctx @ params.W_self_o)             # [D]
            ff = torch.relu(_rms_norm(l_hat, eps=eps) @ params.W_mlp_in).square()
            l_out_t = l_hat + (ff @ params.W_mlp_out)             # [D]
            L_out[b, t] = l_out_t

            # ---------------------
            # Phase III: Write (slot-address routing)
            # ---------------------
            x_write = _rms_norm(l_out_t, eps=eps)                 # [D]
            k_upd = x_write @ params.W_k_write                    # [D]
            v_upd = x_write @ params.W_v_write                    # [D]

            e_slots = _rms_norm(params.E_slots, eps=eps)          # [R, D]
            q_slots = e_slots @ params.W_q_slots_global           # [R, D]

            q_h = q_slots.view(R, n_write_heads, d_h)             # [R, h, d_h]
            k_h = k_upd.view(n_write_heads, d_h)                  # [h, d_h]
            v_h = v_upd.view(n_write_heads, d_h)                  # [h, d_h]

            delta_raw = torch.zeros(R, D, dtype=L_in.dtype, device=L_in.device)
            for j in range(n_write_heads):
                q_slots_j = q_h[:, j, :]                          # [R, d_h]
                logits_w = (q_slots_j @ k_h[j]) / math.sqrt(d_h)  # [R]
                alpha_w = torch.softmax(logits_w, dim=0)          # [R]
                delta_raw[:, j * d_h:(j + 1) * d_h] = alpha_w[:, None] * v_h[j][None, :]

            delta_t = delta_raw @ params.W_o_write                # [R, D]
            delta_all[b, t] = delta_t

            # Sequence-level prefix accumulation with optional segment resets
            if boundary_mask[b, t].item():
                running.zero_()
            running = running + delta_t
            S_curr[b, t] = s_t_prev + running

    return L_out, delta_all, S_curr


# -----------------------------------------------------------------------------
# Decomposed oracle variant (shared context-side factor)
# -----------------------------------------------------------------------------

@dataclass
class GDHDecomposedOracleParams:
    # Shared context-side (long side) factor
    U_ctx_shared: torch.Tensor        # [N, r_ctx]

    # Translator-specific short/context factors
    V_ctx_q_read: torch.Tensor        # [r_ctx, N]
    V_ctx_k_write: torch.Tensor       # [r_ctx, N]
    V_ctx_v_write: torch.Tensor       # [r_ctx, N]

    # Feature-side projections
    W_q_read_short: torch.Tensor      # [D, D]
    W_k_write_short: torch.Tensor     # [D, D]
    W_v_write_short: torch.Tensor     # [D, D]

    # Read globals + mixers/gate
    W_k_read_global: torch.Tensor     # [D, D]
    W_v_read_global: torch.Tensor     # [D, D]
    W_o_read: torch.Tensor            # [D, D]
    W_g_read_mute: torch.Tensor       # [D, 1]
    b_g_read_mute: torch.Tensor       # [1]

    # Process (toy causal transformer)
    W_self_q: torch.Tensor            # [D, D]
    W_self_k: torch.Tensor            # [D, D]
    W_self_v: torch.Tensor            # [D, D]
    W_self_o: torch.Tensor            # [D, D]
    W_mlp_in: torch.Tensor            # [D, D]
    W_mlp_out: torch.Tensor           # [D, D]

    # Write globals + mixer (slot-address routing)
    E_slots: torch.Tensor             # [R, D] permanent learnable slot addresses
    W_q_slots_global: torch.Tensor    # [D, D] projects slot addresses into write queries
    W_o_write: torch.Tensor           # [D, D]


def _causal_row_softmax(m: torch.Tensor) -> torch.Tensor:
    """Apply causal masking and row-softmax to an [N,N] matrix."""
    assert m.ndim == 2 and m.shape[0] == m.shape[1]
    n = m.shape[0]
    mask_future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=m.device), diagonal=1)
    masked = m.masked_fill(mask_future, float("-inf"))
    return torch.softmax(masked, dim=-1)


def make_decomposed_oracle_params(
    *,
    n_seq: int,
    d: int,
    n_slots: int,
    r_ctx: int,
    h: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    zero_init_mixers: bool = False,
) -> GDHDecomposedOracleParams:
    """Create deterministic params for decomposed oracle tests."""
    del h

    torch.manual_seed(seed)

    def rand(*shape, scale=0.02):
        return torch.randn(*shape, device=device, dtype=dtype) * scale

    W_o_read = torch.zeros(d, d, device=device, dtype=dtype) if zero_init_mixers else rand(d, d)
    W_o_write = torch.zeros(d, d, device=device, dtype=dtype) if zero_init_mixers else rand(d, d)

    return GDHDecomposedOracleParams(
        U_ctx_shared=rand(n_seq, r_ctx),
        V_ctx_q_read=rand(r_ctx, n_seq),
        V_ctx_k_write=rand(r_ctx, n_seq),
        V_ctx_v_write=rand(r_ctx, n_seq),
        W_q_read_short=rand(d, d),
        W_k_write_short=rand(d, d),
        W_v_write_short=rand(d, d),
        W_k_read_global=rand(d, d),
        W_v_read_global=rand(d, d),
        W_o_read=W_o_read,
        W_g_read_mute=rand(d, 1),
        b_g_read_mute=torch.zeros(1, device=device, dtype=dtype),
        W_self_q=rand(d, d),
        W_self_k=rand(d, d),
        W_self_v=rand(d, d),
        W_self_o=rand(d, d),
        W_mlp_in=rand(d, d),
        W_mlp_out=rand(d, d),
        E_slots=rand(n_slots, d),
        W_q_slots_global=rand(d, d),
        W_o_write=W_o_write,
    )


def gdh_oracle_layer_decomposed(
    L_in: torch.Tensor,
    S_prev: torch.Tensor,
    params: GDHDecomposedOracleParams,
    *,
    n_write_heads: int,
    boundary_mask: torch.Tensor | None = None,
    context_mixers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Slow reference forward for one GDH layer with decomposed translators.

    Decomposition choice:
    - Shared long/context side: U_ctx_shared [N, r_ctx]
    - Translator-specific short side: V_ctx_* [r_ctx, N]
    - Causal context mixers are built as row-softmax(tril(U @ V_*)).

    If context_mixers=(M_q, M_k, M_v) is provided, those [N,N] causal
    mixers are used directly (bypassing U/V construction). This is useful for
    implementation-equivalence tests.
    """
    assert L_in.ndim == 3, f"L_in must be [B,N,D], got {tuple(L_in.shape)}"
    assert S_prev.ndim == 4, f"S_prev must be [B,N,R,D], got {tuple(S_prev.shape)}"

    B, N, D = L_in.shape
    B2, N2, R, D2 = S_prev.shape
    assert (B, N, D) == (B2, N2, D2), "L_in and S_prev shape mismatch"
    assert D % n_write_heads == 0, "D must be divisible by n_write_heads"

    if boundary_mask is None:
        boundary_mask = torch.zeros(B, N, dtype=torch.bool, device=L_in.device)
    else:
        assert boundary_mask.shape == (B, N), f"boundary_mask must be [B,N], got {tuple(boundary_mask.shape)}"
        boundary_mask = boundary_mask.to(torch.bool)

    d_h = D // n_write_heads

    if context_mixers is None:
        assert params.U_ctx_shared.shape[0] == N, "U_ctx_shared first dim must equal N"
        assert params.V_ctx_q_read.shape[1] == N, "V_ctx_q_read second dim must equal N"
        assert params.V_ctx_k_write.shape[1] == N, "V_ctx_k_write second dim must equal N"
        assert params.V_ctx_v_write.shape[1] == N, "V_ctx_v_write second dim must equal N"

        # Build causal sequence mixers from decomposed context-side factors
        M_q = _causal_row_softmax(params.U_ctx_shared @ params.V_ctx_q_read)   # [N,N]
        M_k = _causal_row_softmax(params.U_ctx_shared @ params.V_ctx_k_write)  # [N,N]
        M_v = _causal_row_softmax(params.U_ctx_shared @ params.V_ctx_v_write)  # [N,N]
    else:
        assert len(context_mixers) == 3, "context_mixers must be (M_q, M_k, M_v)"
        M_q, M_k, M_v = context_mixers
        for name, m in (("M_q", M_q), ("M_k", M_k), ("M_v", M_v)):
            assert m.shape == (N, N), f"{name} must be [N,N], got {tuple(m.shape)}"
            # must be lower-triangular for causality
            assert torch.allclose(m.triu(diagonal=1), torch.zeros_like(m).triu(diagonal=1)), f"{name} must be causal (upper triangle zero)"

    M_q = M_q.to(device=L_in.device, dtype=L_in.dtype)
    M_k = M_k.to(device=L_in.device, dtype=L_in.dtype)
    M_v = M_v.to(device=L_in.device, dtype=L_in.dtype)

    L_out = torch.empty_like(L_in)
    delta_all = torch.zeros(B, N, R, D, dtype=L_in.dtype, device=L_in.device)
    S_curr = torch.empty_like(S_prev)

    for b in range(B):
        running = torch.zeros(R, D, dtype=L_in.dtype, device=L_in.device)

        # Precompute read translator inputs from decomposed context mixer
        x_read_src = _rms_norm(L_in[b], eps=eps)              # [N,D]
        x_read_seq = M_q @ x_read_src                         # [N,D]

        l_tilde_hist: list[torch.Tensor] = []
        l_out_hist: list[torch.Tensor] = []

        for t in range(N):
            l_t = L_in[b, t]          # [D]
            s_t_prev = S_prev[b, t]   # [R,D]

            # ---------------------
            # Phase I: Read
            # ---------------------
            x_read = x_read_seq[t]                                  # [D]
            q_loc = x_read @ params.W_q_read_short                  # [D]

            s_hat = _rms_norm(s_t_prev, eps=eps)                    # [R,D]
            k_mem = s_hat @ params.W_k_read_global                  # [R,D]
            v_mem = s_hat @ params.W_v_read_global                  # [R,D]

            logits_read = (k_mem @ q_loc) / math.sqrt(D)            # [R]
            alpha_read = torch.softmax(logits_read, dim=0)          # [R]
            z_read = alpha_read @ v_mem                             # [D]

            g_read_mute = 0.05 + 0.95 * torch.sigmoid((x_read @ params.W_g_read_mute + params.b_g_read_mute).squeeze(-1))
            l_tilde = l_t + g_read_mute * (z_read @ params.W_o_read)

            # ---------------------
            # Phase II: Process (causal)
            # ---------------------
            l_tilde_hist.append(l_tilde)
            h = torch.stack(l_tilde_hist, dim=0)                    # [t+1,D]
            h_norm = _rms_norm(h, eps=eps)

            q_self = h_norm[-1] @ params.W_self_q                   # [D]
            k_self = h_norm @ params.W_self_k                       # [t+1,D]
            v_self = h_norm @ params.W_self_v                       # [t+1,D]

            logits_self = (k_self @ q_self) / math.sqrt(D)
            alpha_self = torch.softmax(logits_self, dim=0)
            ctx = alpha_self @ v_self                               # [D]

            l_hat = l_tilde + (ctx @ params.W_self_o)
            ff = torch.relu(_rms_norm(l_hat, eps=eps) @ params.W_mlp_in).square()
            l_out_t = l_hat + (ff @ params.W_mlp_out)
            L_out[b, t] = l_out_t
            l_out_hist.append(l_out_t)

            # ---------------------
            # Phase III: Write (decomposed token -> slot-address routing)
            # ---------------------
            out_hist = _rms_norm(torch.stack(l_out_hist, dim=0), eps=eps)   # [t+1,D]
            x_write_k = M_k[t, :t + 1] @ out_hist                            # [D]
            x_write_v = M_v[t, :t + 1] @ out_hist                            # [D]

            e_slots = _rms_norm(params.E_slots, eps=eps)                     # [R,D]
            q_slots = e_slots @ params.W_q_slots_global                       # [R,D]
            k_upd = x_write_k @ params.W_k_write_short                       # [D]
            v_upd = x_write_v @ params.W_v_write_short                       # [D]

            q_h = q_slots.view(R, n_write_heads, d_h)
            k_h = k_upd.view(n_write_heads, d_h)
            v_h = v_upd.view(n_write_heads, d_h)

            delta_raw = torch.zeros(R, D, dtype=L_in.dtype, device=L_in.device)
            for j in range(n_write_heads):
                q_slots_j = q_h[:, j, :]                                     # [R,d_h]
                logits_w = (q_slots_j @ k_h[j]) / math.sqrt(d_h)             # [R]
                alpha_w = torch.softmax(logits_w, dim=0)                     # [R]
                delta_raw[:, j * d_h:(j + 1) * d_h] = alpha_w[:, None] * v_h[j][None, :]

            delta_t = delta_raw @ params.W_o_write
            delta_all[b, t] = delta_t

            if boundary_mask[b, t].item():
                running.zero_()
            running = running + delta_t
            S_curr[b, t] = s_t_prev + running

    return L_out, delta_all, S_curr
