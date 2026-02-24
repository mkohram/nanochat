import pytest
import torch

from nanochat.double_helix import (
    GDHConfig,
    GDHLayer,
    GDHWriteCore,
    _cosine_logit,
    _cosine_similarity,
    validate_gdh_inputs,
)
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


def test_gdh_layer_no_read_gate_mode_ignores_gate_parameters():
    B, N, D, R, h = 1, 5, 8, 3, 2
    local, sidecar_prev = _tiny_inputs(B=B, N=N, D=D, R=R, seed=99)

    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h, use_read_gate=False)
    layer = GDHLayer(cfg)

    out_a = layer.read.forward_sequence(local, sidecar_prev, eps=1e-6)

    with torch.no_grad():
        layer.read.W_g_read.fill_(123.0)
        layer.read.W_g_side.fill_(-456.0)
        layer.read.w_g_interaction.fill_(7.0)
        layer.read.w_g_confidence.fill_(-7.0)
        layer.read.w_g_novelty.fill_(3.0)
        layer.read.w_g_temp.fill_(5.0)
        layer.read.w_g_temp_adv.fill_(-4.0)
        layer.read.w_g_synergy.fill_(2.0)
        layer.read.w_g_querymatch.fill_(9.0)
        layer.read.w_g_queryadv.fill_(-9.0)

    out_b = layer.read.forward_sequence(local, sidecar_prev, eps=1e-6)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


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

    decomp.E_slots = dense.E_slots.clone()
    decomp.W_q_slots_global = dense.W_q_slots_global.clone()
    decomp.W_k_write_short = dense.W_k_write.clone()
    decomp.W_v_write_short = dense.W_v_write.clone()
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


def test_gdh_write_core_loads_legacy_q_write_key():
    d, r = 8, 3
    core = GDHWriteCore(d=d, r=r)
    core.reset_parameters(std=0.02, zero_init_mixer=False)

    sd = core.state_dict()
    legacy_sd = dict(sd)
    legacy_sd["W_q_write_global"] = legacy_sd.pop("W_q_slots_global")

    restored = GDHWriteCore(d=d, r=r)
    restored.load_state_dict(legacy_sd, strict=True)

    assert torch.allclose(restored.W_q_slots_global, core.W_q_slots_global)


def test_gdh_write_eslots_prenorm_is_nearly_scale_invariant_for_positive_rescale():
    d, r, h = 8, 3, 2
    torch.manual_seed(0)

    core_a = GDHWriteCore(d=d, r=r)
    core_a.reset_parameters(std=0.02, zero_init_mixer=False)
    core_b = GDHWriteCore(d=d, r=r)
    core_b.load_state_dict(core_a.state_dict())

    local_out = torch.randn(2, 5, d)
    sidecar_prev = torch.randn(2, 5, r, d)

    delta_a = core_a.forward_sequence_delta(local_out, sidecar_prev, n_write_heads=h, eps=1e-6)

    with torch.no_grad():
        core_b.E_slots.mul_(10.0)

    delta_b = core_b.forward_sequence_delta(local_out, sidecar_prev, n_write_heads=h, eps=1e-6)

    assert torch.allclose(delta_a, delta_b, atol=2e-5, rtol=1e-4)


def _copy_dense_oracle_weights_to_layer(layer: GDHLayer, p):
    with torch.no_grad():
        layer.read.W_q_read.copy_(p.W_q_read)
        layer.read.W_k_read_global.copy_(p.W_k_read_global)
        layer.read.W_v_read_global.copy_(p.W_v_read_global)
        layer.read.W_o_read.copy_(p.W_o_read)
        # New competitive gate reduces to sigmoid(a) when logits are [0, -a].
        layer.read.W_g_read.copy_(-p.W_g_read)
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_temp_adv.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_queryadv2.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
        layer.read.w_g_queryresid.zero_()
        layer.read.w_g_advconfnovel.zero_()
        layer.read.w_g_localquery.zero_()
        layer.read.w_g_localredundancy.zero_()

        layer.process.W_self_q.copy_(p.W_self_q)
        layer.process.W_self_k.copy_(p.W_self_k)
        layer.process.W_self_v.copy_(p.W_self_v)
        layer.process.W_self_o.copy_(p.W_self_o)
        layer.process.W_mlp_in.copy_(p.W_mlp_in)
        layer.process.W_mlp_out.copy_(p.W_mlp_out)

        layer.write.E_slots.copy_(p.E_slots)
        layer.write.W_q_slots_global.copy_(p.W_q_slots_global)
        layer.write.W_k_write.copy_(p.W_k_write)
        layer.write.W_v_write.copy_(p.W_v_write)
        layer.write.W_o_write.copy_(p.W_o_write)


def test_gdh_read_competitive_gate_prefers_sidecar_when_side_logit_wins():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()               # neutral local logit
        layer.read.w_g_interaction.zero_()        # isolate 2-logit competition
        layer.read.w_g_confidence.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar = torch.ones(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.W_g_side.fill_(-4.0)
    out_low, _, _ = layer(local, sidecar, boundary)

    with torch.no_grad():
        layer.read.W_g_side.fill_(4.0)
    out_high, _, _ = layer(local, sidecar, boundary)

    low_delta = (out_low - local).abs().mean().item()
    high_delta = (out_high - local).abs().mean().item()
    assert high_delta > low_delta + 1e-3


def test_gdh_read_interaction_gate_opens_more_when_sidecar_aligns():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        # Isolate read gate behavior.
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.fill_(6.0)
        layer.read.w_g_confidence.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_aligned = torch.ones(B, N, R, D)
    sidecar_anti = -torch.ones(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    out_aligned, _, _ = layer(local, sidecar_aligned, boundary)
    out_anti, _, _ = layer(local, sidecar_anti, boundary)

    # Alignment-sensitive gating should favor sidecar-aligned reads.
    aligned_delta = (out_aligned - local).abs().mean().item()
    anti_delta = (out_anti - local).abs().mean().item()
    assert aligned_delta > anti_delta + 1e-3


def test_gdh_read_confidence_term_can_increase_gate_opening():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar = torch.ones(B, N, R, D)
    sidecar[0, 0, 1] = -torch.ones(D)
    sidecar[0, 0, 2] = -torch.ones(D)
    # One slot aligned with query, others anti-aligned -> low-entropy read attention.

    boundary = torch.zeros(B, N, dtype=torch.bool)
    with torch.no_grad():
        layer.read.w_g_confidence.zero_()
    out_no_conf, _, _ = layer(local, sidecar, boundary)

    with torch.no_grad():
        layer.read.w_g_confidence.fill_(8.0)
    out_with_conf, _, _ = layer(local, sidecar, boundary)

    no_conf_delta = (out_no_conf - local).abs().mean().item()
    with_conf_delta = (out_with_conf - local).abs().mean().item()
    assert with_conf_delta > no_conf_delta + 1e-3


def test_gdh_read_centered_confidence_can_reduce_diffuse_retrieval_opening():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar = torch.ones(B, N, R, D)  # identical slots -> diffuse/uniform read attention
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_confidence.zero_()
    out_no_conf, _, _ = layer(local, sidecar, boundary)

    with torch.no_grad():
        layer.read.w_g_confidence.fill_(8.0)
    out_with_conf, _, _ = layer(local, sidecar, boundary)

    no_conf_delta = (out_no_conf - local).abs().mean().item()
    with_conf_delta = (out_with_conf - local).abs().mean().item()
    assert with_conf_delta < no_conf_delta - 1e-3


def test_gdh_read_temperature_term_sharpens_confident_and_flattens_diffuse_routing():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.fill_(1.0)
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_synergy.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_sharp = torch.ones(B, N, R, D)
    sidecar_sharp[0, 0, 1] = -torch.ones(D)
    sidecar_sharp[0, 0, 2] = -torch.ones(D)
    sidecar_diffuse = torch.ones(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_temp.zero_()
    out_sharp_no_temp, _, _ = layer(local, sidecar_sharp, boundary)
    out_diffuse_no_temp, _, _ = layer(local, sidecar_diffuse, boundary)

    with torch.no_grad():
        layer.read.w_g_temp.fill_(6.0)
    out_sharp_with_temp, _, _ = layer(local, sidecar_sharp, boundary)
    out_diffuse_with_temp, _, _ = layer(local, sidecar_diffuse, boundary)

    sharp_no_temp = (out_sharp_no_temp - local).abs().mean().item()
    diffuse_no_temp = (out_diffuse_no_temp - local).abs().mean().item()
    sharp_with_temp = (out_sharp_with_temp - local).abs().mean().item()
    diffuse_with_temp = (out_diffuse_with_temp - local).abs().mean().item()

    baseline_gap = sharp_no_temp - diffuse_no_temp
    with_temp_gap = sharp_with_temp - diffuse_with_temp
    assert with_temp_gap > baseline_gap + 1e-3


def test_gdh_read_queryadv_temperature_prefers_positive_query_advantage():
    B, N, D, R, h = 1, 1, 8, 2, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        W_q = torch.diag(torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32))
        layer.read.W_q_read.copy_(W_q)
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.fill_(0.5)
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
        layer.read.w_g_advconfnovel.zero_()
        layer.read.w_g_localquery.zero_()
        layer.read.w_g_localredundancy.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    q_vec = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=local.dtype)
    sidecar_pos = torch.zeros(B, N, R, D)
    sidecar_pos[0, 0, 0] = q_vec
    sidecar_pos[0, 0, 1] = -q_vec
    sidecar_neg = torch.zeros(B, N, R, D)
    sidecar_neg[0, 0, 0] = -q_vec
    sidecar_neg[0, 0, 1] = -q_vec
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_temp_adv.zero_()
    out_pos_no_temp, _, _ = layer(local, sidecar_pos, boundary)
    out_neg_no_temp, _, _ = layer(local, sidecar_neg, boundary)

    with torch.no_grad():
        layer.read.w_g_temp_adv.fill_(6.0)
    out_pos_with_temp, _, _ = layer(local, sidecar_pos, boundary)
    out_neg_with_temp, _, _ = layer(local, sidecar_neg, boundary)

    pos_no = (out_pos_no_temp - local).abs().mean().item()
    neg_no = (out_neg_no_temp - local).abs().mean().item()
    pos_yes = (out_pos_with_temp - local).abs().mean().item()
    neg_yes = (out_neg_with_temp - local).abs().mean().item()

    assert pos_yes > pos_no + 1e-3
    assert abs(neg_yes - neg_no) < 1e-3


def test_gdh_read_novelty_term_prefers_nonredundant_sidecar_content():
    B, N, D, R, h = 1, 1, 8, 1, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.fill_(8.0)
        layer.read.w_g_synergy.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_redundant = torch.ones(B, N, R, D)
    orth = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=local.dtype).view(1, 1, 1, D)
    sidecar_novel = orth.clone()
    boundary = torch.zeros(B, N, dtype=torch.bool)

    out_redundant, _, _ = layer(local, sidecar_redundant, boundary)
    out_novel, _, _ = layer(local, sidecar_novel, boundary)

    redundant_delta = (out_redundant - local).abs().mean().item()
    novel_delta = (out_novel - local).abs().mean().item()
    assert novel_delta > redundant_delta + 1e-3


def test_gdh_read_confidence_novelty_synergy_prefers_sharp_novel_retrieval():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.fill_(8.0)

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    # high confidence + high novelty: one orthogonal slot, two anti-aligned distractors.
    sidecar_sharp_novel = torch.zeros(B, N, R, D)
    sidecar_sharp_novel[0, 0, 0] = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=local.dtype)
    sidecar_sharp_novel[0, 0, 1] = -torch.ones(D)
    sidecar_sharp_novel[0, 0, 2] = -torch.ones(D)

    # high confidence + low novelty: one redundant slot, two anti-aligned distractors.
    sidecar_sharp_redundant = torch.zeros(B, N, R, D)
    sidecar_sharp_redundant[0, 0, 0] = torch.ones(D)
    sidecar_sharp_redundant[0, 0, 1] = -torch.ones(D)
    sidecar_sharp_redundant[0, 0, 2] = -torch.ones(D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    out_sharp_novel, _, _ = layer(local, sidecar_sharp_novel, boundary)
    out_sharp_redundant, _, _ = layer(local, sidecar_sharp_redundant, boundary)

    sharp_novel_delta = (out_sharp_novel - local).abs().mean().item()
    sharp_redundant_delta = (out_sharp_redundant - local).abs().mean().item()
    assert sharp_novel_delta > sharp_redundant_delta + 1e-3


def test_gdh_read_querymatch_term_prefers_query_aligned_retrieval():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.fill_(8.0)

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_query_aligned = torch.zeros(B, N, R, D)
    sidecar_query_aligned[0, 0, 0] = torch.ones(D)
    sidecar_query_aligned[0, 0, 1] = -torch.ones(D)
    sidecar_query_aligned[0, 0, 2] = -torch.ones(D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_querymatch.zero_()
    out_no_querymatch, _, _ = layer(local, sidecar_query_aligned, boundary)

    with torch.no_grad():
        layer.read.w_g_querymatch.fill_(8.0)
    out_with_querymatch, _, _ = layer(local, sidecar_query_aligned, boundary)

    no_querymatch_delta = (out_no_querymatch - local).abs().mean().item()
    with_querymatch_delta = (out_with_querymatch - local).abs().mean().item()
    assert with_querymatch_delta > no_querymatch_delta + 1e-3


def test_gdh_read_queryadv_term_boosts_sidecar_when_it_beats_local_query_match():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(-torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_query_better = torch.zeros(B, N, R, D)
    sidecar_query_better[0, 0, 0] = -torch.ones(D)
    sidecar_query_better[0, 0, 1] = torch.ones(D)
    sidecar_query_better[0, 0, 2] = torch.ones(D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_queryadv.zero_()
    out_no_queryadv, _, _ = layer(local, sidecar_query_better, boundary)

    with torch.no_grad():
        layer.read.w_g_queryadv.fill_(30.0)
    out_with_queryadv, _, _ = layer(local, sidecar_query_better, boundary)

    no_queryadv_delta = (out_no_queryadv - local).abs().mean().item()
    with_queryadv_delta = (out_with_queryadv - local).abs().mean().item()
    assert with_queryadv_delta > no_queryadv_delta + 1e-3


def test_gdh_read_queryadv2_term_rewards_decisive_query_advantage_magnitude():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(-torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_strong_adv = torch.zeros(B, N, R, D)
    sidecar_strong_adv[0, 0, 0] = -torch.ones(D)
    sidecar_strong_adv[0, 0, 1] = torch.ones(D)
    sidecar_strong_adv[0, 0, 2] = torch.ones(D)

    sidecar_mild_adv = torch.zeros(B, N, R, D)
    sidecar_mild_adv[0, 0, 0] = torch.cat([-torch.ones(D // 2), torch.ones(D // 2)])
    sidecar_mild_adv[0, 0, 1] = torch.ones(D)
    sidecar_mild_adv[0, 0, 2] = torch.ones(D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_queryadv2.zero_()
    out_strong_no_q2, _, _ = layer(local, sidecar_strong_adv, boundary)
    out_mild_no_q2, _, _ = layer(local, sidecar_mild_adv, boundary)

    with torch.no_grad():
        layer.read.w_g_queryadv2.fill_(20.0)
    out_strong_q2, _, _ = layer(local, sidecar_strong_adv, boundary)
    out_mild_q2, _, _ = layer(local, sidecar_mild_adv, boundary)

    strong_no_q2 = (out_strong_no_q2 - local).abs().mean().item()
    mild_no_q2 = (out_mild_no_q2 - local).abs().mean().item()
    strong_q2 = (out_strong_q2 - local).abs().mean().item()
    mild_q2 = (out_mild_q2 - local).abs().mean().item()

    baseline_gap = strong_no_q2 - mild_no_q2
    with_q2_gap = strong_q2 - mild_q2
    assert with_q2_gap > baseline_gap + 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_queryresid_term_prefers_sidecar_when_local_query_headroom_is_high():
    B, N, D, R, h = 1, 1, 8, 1, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.diag(torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32)))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
        layer.read.w_g_advconfnovel.zero_()
        layer.read.w_g_localquery.zero_()
        layer.read.w_g_localredundancy.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local_high_headroom = torch.ones(B, N, D)
    local_low_headroom = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).view(B, N, D)
    sidecar_high_headroom = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=torch.float32).view(B, N, R, D)
    sidecar_low_headroom = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).view(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_queryresid.zero_()
    out_high_no_term, _, _ = layer(local_high_headroom, sidecar_high_headroom, boundary)
    out_low_no_term, _, _ = layer(local_low_headroom, sidecar_low_headroom, boundary)

    with torch.no_grad():
        layer.read.w_g_queryresid.fill_(10.0)
    out_high_with_term, _, _ = layer(local_high_headroom, sidecar_high_headroom, boundary)
    out_low_with_term, _, _ = layer(local_low_headroom, sidecar_low_headroom, boundary)

    baseline_gap = (out_high_no_term - local_high_headroom).abs().mean().item() - (out_low_no_term - local_low_headroom).abs().mean().item()
    with_term_gap = (out_high_with_term - local_high_headroom).abs().mean().item() - (out_low_with_term - local_low_headroom).abs().mean().item()
    assert with_term_gap > baseline_gap + 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_localquery_term_strengthens_local_branch_when_query_already_local():
    B, N, D, R, h = 1, 1, 8, 3, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
        layer.read.w_g_localredundancy.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    # Retrieval has weak/negative query match while local strongly matches query.
    sidecar_not_query_aligned = torch.zeros(B, N, R, D)
    sidecar_not_query_aligned[0, 0, 0] = -torch.ones(D)
    sidecar_not_query_aligned[0, 0, 1] = -torch.ones(D)
    sidecar_not_query_aligned[0, 0, 2] = -torch.ones(D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_localquery.zero_()
    out_no_localquery, _, _ = layer(local, sidecar_not_query_aligned, boundary)

    with torch.no_grad():
        layer.read.w_g_localquery.fill_(10.0)
    out_with_localquery, _, _ = layer(local, sidecar_not_query_aligned, boundary)

    no_localquery_delta = (out_no_localquery - local).abs().mean().item()
    with_localquery_delta = (out_with_localquery - local).abs().mean().item()
    assert with_localquery_delta < no_localquery_delta - 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_localredundancy_term_prefers_local_when_sidecar_is_redundant():
    B, N, D, R, h = 1, 1, 8, 1, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
        layer.read.w_g_localquery.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_redundant = torch.ones(B, N, R, D)
    sidecar_novel = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=local.dtype).view(1, 1, 1, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_localredundancy.zero_()
    out_redundant_no_term, _, _ = layer(local, sidecar_redundant, boundary)
    out_novel_no_term, _, _ = layer(local, sidecar_novel, boundary)

    with torch.no_grad():
        layer.read.w_g_localredundancy.fill_(12.0)
    out_redundant_with_term, _, _ = layer(local, sidecar_redundant, boundary)
    out_novel_with_term, _, _ = layer(local, sidecar_novel, boundary)

    baseline_gap = (out_novel_no_term - local).abs().mean().item() - (out_redundant_no_term - local).abs().mean().item()
    with_term_gap = (out_novel_with_term - local).abs().mean().item() - (out_redundant_with_term - local).abs().mean().item()
    assert with_term_gap > baseline_gap + 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_advnovel_term_prefers_novel_query_advantage_over_redundant_match():
    B, N, D, R, h = 1, 1, 8, 1, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(-torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    sidecar_redundant = -torch.ones(B, N, R, D)  # query-advantage positive but redundant vs local
    sidecar_novel = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], dtype=local.dtype).view(1, 1, 1, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()
    out_redundant_no_term, _, _ = layer(local, sidecar_redundant, boundary)
    out_novel_no_term, _, _ = layer(local, sidecar_novel, boundary)

    with torch.no_grad():
        layer.read.w_g_advnovel.fill_(12.0)
    out_redundant_with_term, _, _ = layer(local, sidecar_redundant, boundary)
    out_novel_with_term, _, _ = layer(local, sidecar_novel, boundary)

    baseline_gap = (out_novel_no_term - local).abs().mean().item() - (out_redundant_no_term - local).abs().mean().item()
    with_term_gap = (out_novel_with_term - local).abs().mean().item() - (out_redundant_with_term - local).abs().mean().item()
    assert with_term_gap > baseline_gap + 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_advconf_term_prefers_confident_query_advantage_over_diffuse_match():
    B, N, D, R, h = 1, 1, 8, 2, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(-torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconfnovel.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    # Same mean sidecar value in both cases, but different read confidence.
    sidecar_confident = torch.stack([
        -torch.ones(D),
        torch.ones(D),
    ], dim=0).view(1, 1, R, D)
    sidecar_diffuse = -torch.ones(B, N, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_advconf.zero_()
    out_conf_no_term, _, _ = layer(local, sidecar_confident, boundary)
    out_diff_no_term, _, _ = layer(local, sidecar_diffuse, boundary)

    with torch.no_grad():
        layer.read.w_g_advconf.fill_(12.0)
    out_conf_with_term, _, _ = layer(local, sidecar_confident, boundary)
    out_diff_with_term, _, _ = layer(local, sidecar_diffuse, boundary)

    baseline_gap = (out_conf_no_term - local).abs().mean().item() - (out_diff_no_term - local).abs().mean().item()
    with_term_gap = (out_conf_with_term - local).abs().mean().item() - (out_diff_with_term - local).abs().mean().item()
    assert with_term_gap > baseline_gap + 1e-3


@pytest.mark.skip(reason="pruned read-gate experimental terms")
def test_gdh_read_advconfnovel_term_prefers_confident_novel_query_advantage():
    B, N, D, R, h = 1, 1, 8, 2, 2
    cfg = GDHConfig(n_embd=D, n_slots=R, n_write_heads=h)
    layer = GDHLayer(cfg)

    with torch.no_grad():
        layer.read.W_q_read.copy_(-torch.eye(D))
        layer.read.W_k_read_global.copy_(torch.eye(D))
        layer.read.W_v_read_global.copy_(torch.eye(D))
        layer.read.W_o_read.copy_(torch.eye(D))
        layer.read.W_g_read.zero_()
        layer.read.W_g_side.zero_()
        layer.read.w_g_interaction.zero_()
        layer.read.w_g_confidence.zero_()
        layer.read.w_g_novelty.zero_()
        layer.read.w_g_temp.zero_()
        layer.read.w_g_synergy.zero_()
        layer.read.w_g_querymatch.zero_()
        layer.read.w_g_queryadv.zero_()
        layer.read.w_g_advnovel.zero_()
        layer.read.w_g_advconf.zero_()

        layer.process.W_self_q.zero_(); layer.process.W_self_k.zero_(); layer.process.W_self_v.zero_(); layer.process.W_self_o.zero_()
        layer.process.W_mlp_in.zero_(); layer.process.W_mlp_out.zero_()
        layer.write.W_q_slots_global.zero_(); layer.write.W_v_write.zero_(); layer.write.W_k_write.zero_(); layer.write.W_o_write.zero_()

    local = torch.ones(B, N, D)
    # Confident + novel + query-advantage-positive.
    sidecar_confident_novel = torch.stack([
        torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=local.dtype),
        torch.ones(D),
    ], dim=0).view(1, 1, R, D)
    # Confident + redundant + query-advantage-positive.
    sidecar_confident_redundant = torch.stack([
        -torch.ones(D),
        torch.ones(D),
    ], dim=0).view(1, 1, R, D)
    boundary = torch.zeros(B, N, dtype=torch.bool)

    with torch.no_grad():
        layer.read.w_g_advconfnovel.zero_()
    out_novel_no_term, _, _ = layer(local, sidecar_confident_novel, boundary)
    out_redundant_no_term, _, _ = layer(local, sidecar_confident_redundant, boundary)

    with torch.no_grad():
        layer.read.w_g_advconfnovel.fill_(12.0)
    out_novel_with_term, _, _ = layer(local, sidecar_confident_novel, boundary)
    out_redundant_with_term, _, _ = layer(local, sidecar_confident_redundant, boundary)

    baseline_gap = (out_novel_no_term - local).abs().mean().item() - (out_redundant_no_term - local).abs().mean().item()
    with_term_gap = (out_novel_with_term - local).abs().mean().item() - (out_redundant_with_term - local).abs().mean().item()
    assert with_term_gap > baseline_gap + 1e-3


def test_cosine_gate_projection_is_scale_invariant():
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    w = torch.randn(8, 1)

    base = _cosine_logit(x, w)
    x_scaled = _cosine_logit(17.0 * x, w)
    w_scaled = _cosine_logit(x, 0.05 * w)

    assert torch.allclose(base, x_scaled, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base, w_scaled, atol=1e-6, rtol=1e-6)


def test_cosine_similarity_is_scale_invariant_and_bounded():
    torch.manual_seed(7)
    x = torch.randn(3, 8)
    y = torch.randn(3, 8)

    base = _cosine_similarity(x, y)
    x_scaled = _cosine_similarity(19.0 * x, y)
    y_scaled = _cosine_similarity(x, 0.03 * y)

    assert torch.allclose(base, x_scaled, atol=1e-6, rtol=1e-6)
    assert torch.allclose(base, y_scaled, atol=1e-6, rtol=1e-6)
    assert (base <= 1.0 + 1e-6).all()
    assert (base >= -1.0 - 1e-6).all()


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


def test_gpt_gdh_can_disable_read_gate():
    cfg = _tiny_gpt_config(arch="gdh")
    cfg.gdh_use_read_gate = False
    model = GPT(cfg)
    model.init_weights()

    assert all(not read_core.use_read_gate for read_core in model.gdh_read)

    idx = torch.randint(0, cfg.vocab_size, (2, 6), dtype=torch.long)
    logits = model(idx)
    assert logits.shape == (2, 6, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_gpt_gdh_can_disable_write_brain():
    cfg = _tiny_gpt_config(arch="gdh")
    cfg.gdh_use_write_brain = False
    model = GPT(cfg)
    model.init_weights()

    assert all(not write_core.use_write_brain for write_core in model.gdh_write)
    assert all(write_core.W_write_mlp_in_global is None for write_core in model.gdh_write)
    assert all(write_core.W_write_mlp_out_global is None for write_core in model.gdh_write)

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

        # Write-side global brain starts as near-noop residual (safe bootstrap)
        assert torch.count_nonzero(model.gdh_write[0].W_write_mlp_in_global).item() > 0
        assert torch.count_nonzero(model.gdh_write[0].W_write_mlp_out_global).item() == 0


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

    # Slot-address write globals shared across layers
    assert model.gdh_write[0].E_slots is model.gdh_write[1].E_slots
    assert model.gdh_write[1].E_slots is model.gdh_write[2].E_slots
    assert model.gdh_write[0].W_q_slots_global is model.gdh_write[1].W_q_slots_global
    assert model.gdh_write[1].W_q_slots_global is model.gdh_write[2].W_q_slots_global

    # Global write-side sidecar brain is shared across layers
    assert model.gdh_write[0].W_write_mlp_in_global is model.gdh_write[1].W_write_mlp_in_global
    assert model.gdh_write[1].W_write_mlp_in_global is model.gdh_write[2].W_write_mlp_in_global
    assert model.gdh_write[0].W_write_mlp_out_global is model.gdh_write[1].W_write_mlp_out_global
    assert model.gdh_write[1].W_write_mlp_out_global is model.gdh_write[2].W_write_mlp_out_global

    # Local-token write key transform stays layer-specific
    assert model.gdh_write[0].W_k_write is not model.gdh_write[1].W_k_write
    assert model.gdh_write[1].W_k_write is not model.gdh_write[2].W_k_write


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
    local_per_layer = (2 * d * d + 2 * d + 15) + (3 * d * d)  # read-local (+sidecar gate logit + interaction+confidence+novelty+temperature+queryadv-temperature+synergy+querymatch+queryadv+queryadv2+advnovel+advconf+queryresid+advconfnovel+localquery+localredundancy scalars) + write-local
    assert gdh_2 - gdh_1 == local_per_layer


def test_gpt_gdh_global_tying_survives_meta_to_empty_init():
    cfg = _tiny_gpt_config(arch="gdh", n_layer=3)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()

    assert model.gdh_read[0].W_k_read_global is model.gdh_read[1].W_k_read_global
    assert model.gdh_read[1].W_k_read_global is model.gdh_read[2].W_k_read_global

    assert model.gdh_write[0].E_slots is model.gdh_write[1].E_slots
    assert model.gdh_write[1].E_slots is model.gdh_write[2].E_slots
    assert model.gdh_write[0].W_q_slots_global is model.gdh_write[1].W_q_slots_global
    assert model.gdh_write[1].W_q_slots_global is model.gdh_write[2].W_q_slots_global

    assert model.gdh_write[0].W_write_mlp_in_global is model.gdh_write[1].W_write_mlp_in_global
    assert model.gdh_write[1].W_write_mlp_in_global is model.gdh_write[2].W_write_mlp_in_global
    assert model.gdh_write[0].W_write_mlp_out_global is model.gdh_write[1].W_write_mlp_out_global
    assert model.gdh_write[1].W_write_mlp_out_global is model.gdh_write[2].W_write_mlp_out_global

    # Local-token write key transform remains per-layer (not tied)
    assert model.gdh_write[0].W_k_write is not model.gdh_write[1].W_k_write
    assert model.gdh_write[1].W_k_write is not model.gdh_write[2].W_k_write

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
