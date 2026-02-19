import pytest
import torch

from nanochat.double_helix import GDHConfig, GDHLayer, validate_gdh_inputs
from nanochat.gpt import GPT, GPTConfig
from tests.gdh.oracle import (
    make_oracle_params,
    gdh_oracle_layer,
    make_decomposed_oracle_params,
    gdh_oracle_layer_decomposed,
)


def _tiny_inputs(B=2, N=5, D=8, R=3, seed=123):
    torch.manual_seed(seed)
    L = torch.randn(B, N, D)
    S_prev = torch.randn(B, N, R, D)
    return L, S_prev


# -----------------------------------------------------------------------------
# Dense oracle tests
# -----------------------------------------------------------------------------

def test_oracle_shapes_and_finite_values():
    B, N, D, R, h = 2, 5, 8, 3, 2
    L, S_prev = _tiny_inputs(B=B, N=N, D=D, R=R)
    params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=7)

    L_out, delta, S_curr = gdh_oracle_layer(L, S_prev, params, n_write_heads=h)

    assert L_out.shape == (B, N, D)
    assert delta.shape == (B, N, R, D)
    assert S_curr.shape == (B, N, R, D)

    assert torch.isfinite(L_out).all()
    assert torch.isfinite(delta).all()
    assert torch.isfinite(S_curr).all()


def test_zero_init_mixers_disconnect_sidecar_path():
    B, N, D, R, h = 1, 6, 8, 4, 2
    torch.manual_seed(0)

    L = torch.randn(B, N, D)
    S_zero = torch.zeros(B, N, R, D)
    S_rand = torch.randn(B, N, R, D)

    params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=11, zero_init_mixers=True)

    L_out_zero, delta_zero, S_curr_zero = gdh_oracle_layer(L, S_zero, params, n_write_heads=h)
    L_out_rand, delta_rand, S_curr_rand = gdh_oracle_layer(L, S_rand, params, n_write_heads=h)

    assert torch.allclose(L_out_zero, L_out_rand, atol=1e-6, rtol=0)
    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero), atol=0, rtol=0)
    assert torch.allclose(delta_rand, torch.zeros_like(delta_rand), atol=0, rtol=0)
    assert torch.allclose(S_curr_zero, S_zero, atol=0, rtol=0)
    assert torch.allclose(S_curr_rand, S_rand, atol=0, rtol=0)


def test_prefix_accumulation_with_boundary_resets():
    B, N, D, R, h = 1, 6, 8, 3, 2
    torch.manual_seed(1)

    L = torch.randn(B, N, D)
    S_prev = torch.randn(B, N, R, D)
    params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=5)

    boundary = torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.bool)

    _, delta, S_curr = gdh_oracle_layer(L, S_prev, params, n_write_heads=h, boundary_mask=boundary)

    running = torch.zeros(R, D)
    expected = torch.zeros_like(S_curr)
    for t in range(N):
        if boundary[0, t]:
            running.zero_()
        running = running + delta[0, t]
        expected[0, t] = S_prev[0, t] + running

    assert torch.allclose(S_curr, expected, atol=1e-6, rtol=1e-6)


def test_causal_prefix_invariance_on_future_token_change():
    B, N, D, R, h = 1, 7, 8, 3, 2
    torch.manual_seed(3)

    L = torch.randn(B, N, D)
    S_prev = torch.zeros(B, N, R, D)
    params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=17)

    L_mod = L.clone()
    L_mod[0, -1] = L_mod[0, -1] + 10.0

    L_out_a, delta_a, S_curr_a = gdh_oracle_layer(L, S_prev, params, n_write_heads=h)
    L_out_b, delta_b, S_curr_b = gdh_oracle_layer(L_mod, S_prev, params, n_write_heads=h)

    assert torch.allclose(L_out_a[:, :-1], L_out_b[:, :-1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(delta_a[:, :-1], delta_b[:, :-1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(S_curr_a[:, :-1], S_curr_b[:, :-1], atol=1e-6, rtol=1e-6)


def test_past_token_change_affects_future_deltas():
    B, N, D, R, h = 1, 8, 8, 3, 2
    torch.manual_seed(123)

    L = torch.randn(B, N, D)
    S_prev = torch.zeros(B, N, R, D)
    params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=17)

    params.W_self_q *= 10
    params.W_self_k *= 10
    params.W_self_v *= 10
    params.W_self_o *= 10
    params.W_mlp_in *= 10
    params.W_mlp_out *= 10
    params.W_k_write *= 10
    params.W_v_write *= 10
    params.W_o_write *= 10

    L_out_a, delta_a, _ = gdh_oracle_layer(L, S_prev, params, n_write_heads=h)

    t_change = 2
    L_mod = L.clone()
    L_mod[0, t_change] += 5.0
    L_out_b, delta_b, _ = gdh_oracle_layer(L_mod, S_prev, params, n_write_heads=h)

    assert torch.allclose(delta_a[:, :t_change], delta_b[:, :t_change], atol=1e-6, rtol=1e-6)
    assert torch.allclose(L_out_a[:, :t_change], L_out_b[:, :t_change], atol=1e-6, rtol=1e-6)

    future_diff = (delta_a[:, t_change + 1:] - delta_b[:, t_change + 1:]).abs().max().item()
    assert future_diff > 1e-4, f"Expected future delta change, got max diff={future_diff:.6g}"


def test_invalid_head_count_raises():
    B, N, D, R = 1, 4, 8, 3
    L, S_prev = _tiny_inputs(B=B, N=N, D=D, R=R, seed=9)
    params = make_oracle_params(d=D, n_slots=R, r=2, h=2, seed=9)

    with pytest.raises(AssertionError, match="divisible"):
        gdh_oracle_layer(L, S_prev, params, n_write_heads=3)


def test_read_gate_strength_controls_injection_magnitude():
    B, N, D, R, h = 1, 5, 8, 3, 2
    torch.manual_seed(33)

    L = torch.rand(B, N, D)
    S_prev = torch.rand(B, N, R, D)

    params_off = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=101)
    params_off.W_self_q.zero_(); params_off.W_self_k.zero_(); params_off.W_self_v.zero_(); params_off.W_self_o.zero_()
    params_off.W_mlp_in.zero_(); params_off.W_mlp_out.zero_(); params_off.W_o_write.zero_()
    params_off.W_g_read.fill_(-20.0)

    params_on = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=101)
    params_on.W_self_q.zero_(); params_on.W_self_k.zero_(); params_on.W_self_v.zero_(); params_on.W_self_o.zero_()
    params_on.W_mlp_in.zero_(); params_on.W_mlp_out.zero_(); params_on.W_o_write.zero_()
    params_on.W_g_read.fill_(20.0)

    L_out_off, _, _ = gdh_oracle_layer(L, S_prev, params_off, n_write_heads=h)
    L_out_on, _, _ = gdh_oracle_layer(L, S_prev, params_on, n_write_heads=h)

    off_mag = (L_out_off - L).abs().mean().item()
    on_mag = (L_out_on - L).abs().mean().item()

    assert on_mag > off_mag + 1e-5, f"Expected stronger gate to inject more, got on={on_mag:.6g}, off={off_mag:.6g}"


# -----------------------------------------------------------------------------
# Decomposed oracle tests
# -----------------------------------------------------------------------------

def test_decomposed_oracle_shapes_and_finite_values():
    B, N, D, R, h = 2, 6, 8, 3, 2
    L, S_prev = _tiny_inputs(B=B, N=N, D=D, R=R)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=7)

    L_out, delta, S_curr = gdh_oracle_layer_decomposed(L, S_prev, params, n_write_heads=h)

    assert L_out.shape == (B, N, D)
    assert delta.shape == (B, N, R, D)
    assert S_curr.shape == (B, N, R, D)

    assert torch.isfinite(L_out).all()
    assert torch.isfinite(delta).all()
    assert torch.isfinite(S_curr).all()


def test_decomposed_zero_init_mixers_disconnect_sidecar_path():
    B, N, D, R, h = 1, 6, 8, 4, 2
    torch.manual_seed(0)

    L = torch.randn(B, N, D)
    S_zero = torch.zeros(B, N, R, D)
    S_rand = torch.randn(B, N, R, D)

    params = make_decomposed_oracle_params(
        n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=11, zero_init_mixers=True
    )

    L_out_zero, delta_zero, S_curr_zero = gdh_oracle_layer_decomposed(L, S_zero, params, n_write_heads=h)
    L_out_rand, delta_rand, S_curr_rand = gdh_oracle_layer_decomposed(L, S_rand, params, n_write_heads=h)

    assert torch.allclose(L_out_zero, L_out_rand, atol=1e-6, rtol=0)
    assert torch.allclose(delta_zero, torch.zeros_like(delta_zero), atol=0, rtol=0)
    assert torch.allclose(delta_rand, torch.zeros_like(delta_rand), atol=0, rtol=0)
    assert torch.allclose(S_curr_zero, S_zero, atol=0, rtol=0)
    assert torch.allclose(S_curr_rand, S_rand, atol=0, rtol=0)


def test_decomposed_causal_prefix_invariance_on_future_token_change():
    B, N, D, R, h = 1, 7, 8, 3, 2
    torch.manual_seed(3)

    L = torch.randn(B, N, D)
    S_prev = torch.zeros(B, N, R, D)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=17)

    L_mod = L.clone()
    L_mod[0, -1] += 10.0

    L_out_a, delta_a, S_curr_a = gdh_oracle_layer_decomposed(L, S_prev, params, n_write_heads=h)
    L_out_b, delta_b, S_curr_b = gdh_oracle_layer_decomposed(L_mod, S_prev, params, n_write_heads=h)

    assert torch.allclose(L_out_a[:, :-1], L_out_b[:, :-1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(delta_a[:, :-1], delta_b[:, :-1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(S_curr_a[:, :-1], S_curr_b[:, :-1], atol=1e-6, rtol=1e-6)


def test_decomposed_past_token_change_affects_future_deltas():
    B, N, D, R, h = 1, 8, 8, 3, 2
    torch.manual_seed(123)

    L = torch.randn(B, N, D)
    S_prev = torch.zeros(B, N, R, D)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=17)

    params.W_self_q *= 10
    params.W_self_k *= 10
    params.W_self_v *= 10
    params.W_self_o *= 10
    params.W_mlp_in *= 10
    params.W_mlp_out *= 10
    params.W_k_write_short *= 10
    params.W_v_write_short *= 10
    params.W_o_write *= 10

    L_out_a, delta_a, _ = gdh_oracle_layer_decomposed(L, S_prev, params, n_write_heads=h)

    t_change = 2
    L_mod = L.clone()
    L_mod[0, t_change] += 5.0

    L_out_b, delta_b, _ = gdh_oracle_layer_decomposed(L_mod, S_prev, params, n_write_heads=h)

    assert torch.allclose(delta_a[:, :t_change], delta_b[:, :t_change], atol=1e-6, rtol=1e-6)
    assert torch.allclose(L_out_a[:, :t_change], L_out_b[:, :t_change], atol=1e-6, rtol=1e-6)

    future_diff = (delta_a[:, t_change + 1:] - delta_b[:, t_change + 1:]).abs().max().item()
    assert future_diff > 1e-4, f"Expected future delta change, got max diff={future_diff:.6g}"


def test_decomposed_prefix_accumulation_with_boundary_resets():
    B, N, D, R, h = 1, 6, 8, 3, 2
    torch.manual_seed(1)

    L = torch.randn(B, N, D)
    S_prev = torch.randn(B, N, R, D)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=5)

    boundary = torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.bool)

    _, delta, S_curr = gdh_oracle_layer_decomposed(
        L, S_prev, params, n_write_heads=h, boundary_mask=boundary
    )

    running = torch.zeros(R, D)
    expected = torch.zeros_like(S_curr)
    for t in range(N):
        if boundary[0, t]:
            running.zero_()
        running = running + delta[0, t]
        expected[0, t] = S_prev[0, t] + running

    assert torch.allclose(S_curr, expected, atol=1e-6, rtol=1e-6)


def test_decomposed_invalid_head_count_raises():
    B, N, D, R = 1, 4, 8, 3
    L, S_prev = _tiny_inputs(B=B, N=N, D=D, R=R, seed=22)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=2, seed=22)

    with pytest.raises(AssertionError, match="divisible"):
        gdh_oracle_layer_decomposed(L, S_prev, params, n_write_heads=3)


def test_decomposed_context_mixers_must_be_causal_when_overridden():
    B, N, D, R, h = 1, 5, 8, 3, 2
    L, S_prev = _tiny_inputs(B=B, N=N, D=D, R=R, seed=31)
    params = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=31)

    M_q = torch.eye(N)
    M_k = torch.eye(N)
    M_v = torch.eye(N)
    M_q[0, 1] = 0.1

    with pytest.raises(AssertionError, match="causal"):
        gdh_oracle_layer_decomposed(
            L,
            S_prev,
            params,
            n_write_heads=h,
            context_mixers=(M_q, M_k, M_v),
        )


# -----------------------------------------------------------------------------
# Cross-oracle comparison tests
# -----------------------------------------------------------------------------

def _copy_dense_weights_into_decomposed(dense, decomp):
    decomp.W_q_read_short = dense.W_q_read.clone()
    decomp.W_k_read_global = dense.W_k_read_global.clone()
    decomp.W_v_read_global = dense.W_v_read_global.clone()
    decomp.W_o_read = dense.W_o_read.clone()
    decomp.W_g_read = dense.W_g_read.clone()

    decomp.W_self_q = dense.W_self_q.clone()
    decomp.W_self_k = dense.W_self_k.clone()
    decomp.W_self_v = dense.W_self_v.clone()
    decomp.W_self_o = dense.W_self_o.clone()
    decomp.W_mlp_in = dense.W_mlp_in.clone()
    decomp.W_mlp_out = dense.W_mlp_out.clone()

    decomp.W_k_write_short = dense.W_k_write.clone()
    decomp.W_v_write_short = dense.W_v_write.clone()
    decomp.W_q_write_global = dense.W_q_write_global.clone()
    decomp.W_o_write = dense.W_o_write.clone()


def test_dense_vs_decomposed_equivalence_with_identity_context_mixers():
    B, N, D, R, h = 2, 6, 8, 3, 2

    torch.manual_seed(123)
    L = torch.randn(B, N, D)
    S_prev = torch.randn(B, N, R, D)
    boundary = torch.tensor([[1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=torch.bool)

    dense = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=11)
    decomp = make_decomposed_oracle_params(n_seq=N, d=D, n_slots=R, r_ctx=3, h=h, seed=99)
    _copy_dense_weights_into_decomposed(dense, decomp)

    I = torch.eye(N, dtype=L.dtype)
    context_mixers = (I, I, I)

    L_out_a, delta_a, S_curr_a = gdh_oracle_layer(
        L, S_prev, dense, n_write_heads=h, boundary_mask=boundary
    )
    L_out_b, delta_b, S_curr_b = gdh_oracle_layer_decomposed(
        L,
        S_prev,
        decomp,
        n_write_heads=h,
        boundary_mask=boundary,
        context_mixers=context_mixers,
    )

    assert torch.allclose(L_out_a, L_out_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(delta_a, delta_b, atol=1e-6, rtol=1e-6)
    assert torch.allclose(S_curr_a, S_curr_b, atol=1e-6, rtol=1e-6)


# -----------------------------------------------------------------------------
# GDH module scaffold tests (single implementation file contract)
# -----------------------------------------------------------------------------

def test_gdh_config_validates_divisibility():
    cfg = GDHConfig(n_embd=16, n_slots=4, n_write_heads=4)
    cfg.validate()

    with pytest.raises(ValueError, match="divisible"):
        GDHConfig(n_embd=10, n_slots=4, n_write_heads=4).validate()


def test_validate_gdh_inputs_contract_checks():
    cfg = GDHConfig(n_embd=8, n_slots=3, n_write_heads=2)

    local = torch.randn(2, 5, 8)
    sidecar = torch.randn(2, 5, 3, 8)
    boundary = torch.zeros(2, 5, dtype=torch.bool)

    validate_gdh_inputs(local, sidecar, cfg, boundary)

    with pytest.raises(ValueError, match="boundary_mask"):
        validate_gdh_inputs(local, sidecar, cfg, torch.zeros(2, 4, dtype=torch.bool))

    with pytest.raises(ValueError, match="sidecar R"):
        validate_gdh_inputs(local, torch.randn(2, 5, 4, 8), cfg, boundary)


def _copy_dense_oracle_weights_to_layer(layer: GDHLayer, p):
    with torch.no_grad():
        layer.read.W_q_read.copy_(p.W_q_read)
        layer.read.W_k_read_global.copy_(p.W_k_read_global)
        layer.read.W_v_read_global.copy_(p.W_v_read_global)
        layer.read.W_o_read.copy_(p.W_o_read)
        layer.read.W_g_read.copy_(p.W_g_read)

        layer.process.W_self_q.copy_(p.W_self_q)
        layer.process.W_self_k.copy_(p.W_self_k)
        layer.process.W_self_v.copy_(p.W_self_v)
        layer.process.W_self_o.copy_(p.W_self_o)
        layer.process.W_mlp_in.copy_(p.W_mlp_in)
        layer.process.W_mlp_out.copy_(p.W_mlp_out)

        layer.write.W_k_write.copy_(p.W_k_write)
        layer.write.W_v_write.copy_(p.W_v_write)
        layer.write.W_q_write_global.copy_(p.W_q_write_global)
        layer.write.W_o_write.copy_(p.W_o_write)


def test_gdh_layer_forward_shapes_and_finite_values():
    B, N, D, R, h = 2, 5, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    local = torch.randn(B, N, D)
    sidecar = torch.randn(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    local_out, delta, sidecar_curr = layer(local, sidecar, boundary)

    assert local_out.shape == (B, N, D)
    assert delta.shape == (B, N, R, D)
    assert sidecar_curr.shape == (B, N, R, D)

    assert torch.isfinite(local_out).all()
    assert torch.isfinite(delta).all()
    assert torch.isfinite(sidecar_curr).all()


def test_gdh_layer_matches_dense_oracle():
    B, N, D, R, h = 2, 6, 8, 3, 2
    torch.manual_seed(123)

    local = torch.randn(B, N, D)
    sidecar_prev = torch.randn(B, N, R, D)
    boundary = torch.tensor([[1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=torch.bool)

    oracle_params = make_oracle_params(d=D, n_slots=R, r=2, h=h, seed=11)
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)
    _copy_dense_oracle_weights_to_layer(layer, oracle_params)

    L_ref, delta_ref, S_ref = gdh_oracle_layer(
        local, sidecar_prev, oracle_params, n_write_heads=h, boundary_mask=boundary
    )
    L_impl, delta_impl, S_impl = layer(local, sidecar_prev, boundary)

    assert torch.allclose(L_ref, L_impl, atol=1e-6, rtol=1e-6)
    assert torch.allclose(delta_ref, delta_impl, atol=1e-6, rtol=1e-6)
    assert torch.allclose(S_ref, S_impl, atol=1e-6, rtol=1e-6)


# -----------------------------------------------------------------------------
# GPT integration sanity tests (minimal, baseline path untouched)
# -----------------------------------------------------------------------------

def _tiny_gpt_config(*, arch: str, n_layer: int = 1):
    return GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch=arch,
        gdh_slots=3,
        gdh_write_heads=4,
    )


def test_gpt_gdh_forward_runs_no_kvcache():
    cfg = _tiny_gpt_config(arch="gdh")
    model = GPT(cfg)
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    logits = model(idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_gpt_gdh_noop_mixers_match_baseline_output():
    torch.manual_seed(123)
    cfg_base = _tiny_gpt_config(arch="baseline")
    cfg_gdh = _tiny_gpt_config(arch="gdh")

    model_base = GPT(cfg_base)
    model_base.init_weights()

    model_gdh = GPT(cfg_gdh)
    model_gdh.init_weights()  # initialize GDH params to finite values
    # copy shared baseline weights
    model_gdh.load_state_dict(model_base.state_dict(), strict=False)

    # force GDH wrappers to no-op
    with torch.no_grad():
        for i in range(cfg_gdh.n_layer):
            model_gdh.gdh_read[i].W_o_read.zero_()
            model_gdh.gdh_write[i].W_o_write.zero_()

    idx = torch.randint(0, cfg_base.vocab_size, (2, 6), dtype=torch.long)

    logits_base = model_base(idx)
    logits_gdh = model_gdh(idx)

    assert torch.allclose(logits_base, logits_gdh, atol=1e-6, rtol=1e-6)


def test_gpt_gdh_option1_init_bootstrap_contract():
    cfg = _tiny_gpt_config(arch="gdh", n_layer=2)
    model = GPT(cfg)
    model.init_weights()

    with torch.no_grad():
        for i in range(cfg.n_layer):
            # Option-1 contract: read mixer starts at 0, write mixer starts non-zero.
            assert torch.count_nonzero(model.gdh_read[i].W_o_read).item() == 0
            assert torch.count_nonzero(model.gdh_write[i].W_o_write).item() > 0


def test_gpt_gdh_depth2_has_nonzero_gdh_grads_at_init():
    torch.manual_seed(123)
    cfg = _tiny_gpt_config(arch="gdh", n_layer=2)
    model = GPT(cfg)
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)

    loss = model(idx, targets)
    loss.backward()

    gdh_grad_norms = []
    for name, p in model.named_parameters():
        if "gdh_" in name:
            g = 0.0 if p.grad is None else p.grad.norm().item()
            gdh_grad_norms.append(g)

    assert any(g > 0 for g in gdh_grad_norms), "Expected at least one GDH gradient at init for n_layer=2"


def test_gpt_gdh_global_weights_are_tied_across_layers():
    cfg = _tiny_gpt_config(arch="gdh", n_layer=3)
    model = GPT(cfg)

    # Read globals shared across layers
    assert model.gdh_read[0].W_k_read_global is model.gdh_read[1].W_k_read_global
    assert model.gdh_read[1].W_k_read_global is model.gdh_read[2].W_k_read_global

    assert model.gdh_read[0].W_v_read_global is model.gdh_read[1].W_v_read_global
    assert model.gdh_read[1].W_v_read_global is model.gdh_read[2].W_v_read_global

    # Write global shared across layers
    assert model.gdh_write[0].W_q_write_global is model.gdh_write[1].W_q_write_global
    assert model.gdh_write[1].W_q_write_global is model.gdh_write[2].W_q_write_global


def test_gpt_gdh_global_weight_count_not_multiplied_by_layers():
    cfg1 = _tiny_gpt_config(arch="gdh", n_layer=1)
    cfg2 = _tiny_gpt_config(arch="gdh", n_layer=2)

    m1 = GPT(cfg1)
    m2 = GPT(cfg2)
    m1.init_weights()
    m2.init_weights()

    gdh_1 = m1.num_scaling_params()["gdh_matrices"]
    gdh_2 = m2.num_scaling_params()["gdh_matrices"]

    d = cfg1.n_embd
    local_per_layer = (2 * d * d + d) + (3 * d * d)  # read-local + write-local
    assert gdh_2 - gdh_1 == local_per_layer


def test_gpt_gdh_global_tying_survives_meta_to_empty_init():
    cfg = _tiny_gpt_config(arch="gdh", n_layer=3)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()

    assert model.gdh_read[0].W_k_read_global is model.gdh_read[1].W_k_read_global
    assert model.gdh_read[1].W_k_read_global is model.gdh_read[2].W_k_read_global
    assert model.gdh_write[0].W_q_write_global is model.gdh_write[1].W_q_write_global
    assert model.gdh_write[1].W_q_write_global is model.gdh_write[2].W_q_write_global

    # Should not trigger the internal parameter-count assertion
    _ = model.num_scaling_params()


def test_gpt_baseline_forward_deterministic_same_weights():
    cfg = _tiny_gpt_config(arch="baseline")
    model_a = GPT(cfg)
    model_b = GPT(cfg)

    torch.manual_seed(123)
    model_a.init_weights()
    torch.manual_seed(123)
    model_b.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    out_a = model_a(idx)
    out_b = model_b(idx)

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_setup_optimizer_includes_gdh_params_only_in_gdh_arch():
    cfg_base = _tiny_gpt_config(arch="baseline")
    cfg_gdh = _tiny_gpt_config(arch="gdh")

    model_base = GPT(cfg_base)
    model_base.init_weights()
    model_gdh = GPT(cfg_gdh)
    model_gdh.init_weights()

    n_base = model_base.num_scaling_params()["total"]
    n_gdh = model_gdh.num_scaling_params()["total"]
    gdh_extra = model_gdh.num_scaling_params().get("gdh_matrices", 0)

    assert n_gdh > n_base
    assert gdh_extra > 0

    # Optimizer should build successfully for both
    opt_base = model_base.setup_optimizer()
    opt_gdh = model_gdh.setup_optimizer()
    assert len(opt_base.param_groups) > 0
    assert len(opt_gdh.param_groups) > 0


def test_setup_optimizer_routes_gdh_params_to_adamw():
    cfg = _tiny_gpt_config(arch="gdh")
    model = GPT(cfg)
    model.init_weights()
    opt = model.setup_optimizer()

    gdh_param_ids = {id(p) for p in model.gdh_read.parameters()} | {id(p) for p in model.gdh_write.parameters()}
    assert len(gdh_param_ids) > 0

    adamw_ids = set()
    muon_ids = set()
    for group in opt.param_groups:
        ids = {id(p) for p in group["params"]}
        if group["kind"] == "adamw":
            adamw_ids |= ids
        elif group["kind"] == "muon":
            muon_ids |= ids

    assert gdh_param_ids.issubset(adamw_ids)
    assert gdh_param_ids.isdisjoint(muon_ids)


def test_gpt_gdh_kvcache_guardrail_raises():
    cfg = _tiny_gpt_config(arch="gdh")
    model = GPT(cfg)
    model.init_weights()

    class _DummyCache:
        def get_pos(self):
            return 0

    idx = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    with pytest.raises(NotImplementedError, match="kv_cache"):
        model(idx, kv_cache=_DummyCache())


def test_gpt_gdh_backward_smoke():
    cfg = _tiny_gpt_config(arch="gdh")
    model = GPT(cfg)
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)

    loss = model(idx, targets)
    assert torch.isfinite(loss).item()

    loss.backward()

    any_grad = False
    for p in model.parameters():
        if p.grad is not None:
            any_grad = True
            assert torch.isfinite(p.grad).all().item()

    assert any_grad


def test_gpt_gdh_state_dict_roundtrip():
    cfg = _tiny_gpt_config(arch="gdh")

    torch.manual_seed(123)
    model_a = GPT(cfg)
    model_a.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    out_a = model_a(idx)

    state = model_a.state_dict()

    model_b = GPT(cfg)
    model_b.load_state_dict(state)
    out_b = model_b(idx)

    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)
