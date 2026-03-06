import torch

from nanochat.gpt import GPT, GPTConfig


def test_final_delta_tanh_bounds_range():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    x = torch.randn(2, 8, cfg.n_embd)
    delta, _, _ = model._gdh_write_probe_delta(
        model.gdh_write[0],
        x,
        gdh_heads=cfg.gdh_write_heads,
        route_topk=0,
        gate_proj=model.gdh_write_gate[0],
        eps=1e-6,
    )
    delta_final = torch.tanh(delta)
    assert delta_final.abs().max().item() <= 1.0001


def test_ema_scan_with_tanh_delta_stays_bounded():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    B, T, R, D = 2, 128, cfg.gdh_slots, cfg.n_embd
    torch.manual_seed(0)
    # Final-bounded deltas in [-1,1]
    delta = torch.tanh(torch.randn(B, T, R, D) * 8.0)
    # Data-dependent retention from gate-like logits
    retention = torch.sigmoid(torch.randn(B, T, 1, 1) * 3.0)

    state = model._gdh_scan_accumulate_ema(delta, retention, boundary_mask=None)
    assert state.abs().max().item() <= 1.0001


def test_ema_scan_no_nan_with_extreme_retention_values():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    B, T, R, D = 2, 256, cfg.gdh_slots, cfg.n_embd
    torch.manual_seed(1)
    delta = torch.tanh(torch.randn(B, T, R, D) * 20.0)
    # Push near-extreme gate values to stress numerical stability.
    retention = torch.sigmoid(torch.randn(B, T, 1, 1) * 20.0)
    boundary = torch.zeros(B, T, dtype=torch.bool)
    boundary[:, ::64] = True

    state = model._gdh_scan_accumulate_ema(delta, retention, boundary_mask=boundary)
    assert torch.isfinite(state).all()
    assert state.abs().max().item() <= 1.0001


def test_slotwise_write_gate_shapes_and_delta_application():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch="gdh",
        gdh_slots=6,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    x = torch.randn(2, 10, cfg.n_embd)
    delta, alpha_soft, g_write = model._gdh_write_probe_delta(
        model.gdh_write[0],
        x,
        gdh_heads=cfg.gdh_write_heads,
        route_topk=0,
        gate_proj=model.gdh_write_gate[0],
        eps=1e-6,
    )
    assert alpha_soft.shape == (2, 10, cfg.gdh_write_heads, cfg.gdh_slots)
    assert delta.shape == (2, 10, cfg.gdh_slots, cfg.n_embd)
    assert g_write is not None
    assert g_write.shape == (2, 10, cfg.gdh_slots, 1)


def test_ema_scan_accepts_slotwise_retention_shape():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        arch="gdh",
        gdh_slots=5,
        gdh_write_heads=4,
    )
    model = GPT(cfg)
    model.init_weights()

    B, T, R, D = 2, 64, cfg.gdh_slots, cfg.n_embd
    delta = torch.tanh(torch.randn(B, T, R, D) * 6.0)
    retention_slotwise = torch.sigmoid(torch.randn(B, T, R, 1) * 2.0)
    boundary = torch.zeros(B, T, dtype=torch.bool)
    boundary[:, ::17] = True
    state = model._gdh_scan_accumulate_ema(delta, retention_slotwise, boundary_mask=boundary)
    assert state.shape == delta.shape
    assert torch.isfinite(state).all()


def test_read_mute_gate_init_bias_is_quiet():
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
    )
    model = GPT(cfg)
    model.init_weights()
    assert abs(float(model.gdh_read[0].b_g_read_mute.item()) - (-1.0)) < 1e-6


def test_ema_write_gate_polarity_retention_not_prescaled():
    # In EMA mode, gate output is retention g; delta should not be pre-multiplied by g.
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()
    x = torch.randn(1, 4, cfg.n_embd)
    # Force retention gate near 1.0
    with torch.no_grad():
        model.gdh_write_gate[0].weight.zero_()
        model.gdh_write_gate[0].bias.fill_(10.0)

    delta_gated, _, g = model._gdh_write_probe_delta(
        model.gdh_write[0], x, gdh_heads=cfg.gdh_write_heads, route_topk=0, gate_proj=model.gdh_write_gate[0], eps=1e-6
    )
    delta_nogate, _, _ = model._gdh_write_probe_delta(
        model.gdh_write[0], x, gdh_heads=cfg.gdh_write_heads, route_topk=0, gate_proj=None, eps=1e-6
    )
    assert g is not None and g.mean().item() > 0.99
    assert torch.allclose(delta_gated, delta_nogate, atol=1e-6, rtol=1e-6)


def test_read_mute_gate_floor_does_not_force_nonzero_output_when_read_mixer_zeroed():
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
    )
    model = GPT(cfg)
    model.init_weights()

    read = model.gdh_read[0]
    with torch.no_grad():
        # Force logits to be very negative so floor behavior dominates.
        read.W_g_read_mute.fill_(-50.0)

    # Use positive inputs so strongly negative weights push sigmoid(logit) -> 0.
    x = torch.ones(2, 6, cfg.n_embd)
    sidecar = torch.randn(2, 6, cfg.gdh_slots, cfg.n_embd)
    y = read.forward_sequence(x, sidecar, eps=1e-6)
    # v2.3 ReZero: with W_o_read=0 at init, outward read coupling is exactly silent.
    assert torch.allclose(y, x, atol=1e-4, rtol=1e-4)


def test_segmented_leaky_scan_no_boundary_matches_reference():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=3,
        gdh_write_heads=4,
    )
    model = GPT(cfg)

    torch.manual_seed(0)
    delta = torch.randn(2, 20, cfg.gdh_slots, cfg.n_embd)
    boundary = torch.zeros(2, 20, dtype=torch.bool)
    beta = 0.9

    got = model._gdh_scan_accumulate(delta, beta=beta, boundary_mask=boundary)

    # Reference leaky scan with implicit segment start at t=0.
    ref = torch.empty_like(delta)
    state = torch.zeros_like(delta[:, 0])
    for t in range(delta.shape[1]):
        state = state * beta + delta[:, t]
        ref[:, t] = state

    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)


def test_write_gate_projection_width_matches_config():
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=7,
        gdh_use_write_gate=True,
    )
    m = GPT(cfg)
    # Canonical path uses slot-wise write gate only.
    assert m.gdh_write_gate[0].out_features == cfg.gdh_slots


def test_pre_tanh_rmsnorm_reduces_saturation_pressure():
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=False,
    )
    model = GPT(cfg)
    model.init_weights()

    torch.manual_seed(0)
    raw = torch.randn(2, 6, cfg.gdh_slots, cfg.n_embd) * 50.0
    out_plain = torch.tanh(raw)
    out_norm = torch.tanh(torch.nn.functional.rms_norm(raw, raw.shape[-1:]))

    # RMSNorm before tanh should keep outputs away from hard rails more than plain tanh.
    sat_plain = (out_plain.abs() > 0.95).float().mean().item()
    sat_norm = (out_norm.abs() > 0.95).float().mean().item()
    assert sat_norm < sat_plain


def test_ema_mode_write_gate_is_retention_not_delta_scaler():
    # Canonical EMA path: gate output is retention g, delta itself is not pre-scaled by g.
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        n_embd=16,
        window_pattern="L",
        arch="gdh",
        gdh_slots=4,
        gdh_write_heads=4,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    x = torch.randn(1, 4, cfg.n_embd)
    with torch.no_grad():
        model.gdh_write_gate[0].weight.zero_()
        model.gdh_write_gate[0].bias.fill_(-12.0)  # retention ~ 0

    delta_gated, _, g = model._gdh_write_probe_delta(
        model.gdh_write[0], x, gdh_heads=cfg.gdh_write_heads, route_topk=0, gate_proj=model.gdh_write_gate[0], eps=1e-6
    )
    delta_nogate, _, _ = model._gdh_write_probe_delta(
        model.gdh_write[0], x, gdh_heads=cfg.gdh_write_heads, route_topk=0, gate_proj=None, eps=1e-6
    )
    assert g is not None and g.mean().item() < 1e-4
    assert torch.allclose(delta_gated, delta_nogate, atol=1e-6, rtol=1e-6)


def test_gdh_global_params_use_full_lr_group_v22():
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
        arch="gdh",
        gdh_slots=8,
        gdh_write_heads=4,
        gdh_use_write_brain=True,
        gdh_write_brain_hidden_mult=1,
        gdh_use_write_gate=True,
    )
    model = GPT(cfg)
    model.init_weights()

    matrix_lr = 0.02
    opt = model.setup_optimizer(matrix_lr=matrix_lr)

    global_targets = {
        id(model.gdh_read[0].W_k_read_global),
        id(model.gdh_read[0].W_v_read_global),
        id(model.gdh_write[0].E_slots),
        id(model.gdh_write[0].W_q_slots_global),
    }
    if model.gdh_write[0].W_write_mlp_in_global is not None:
        global_targets |= {
            id(model.gdh_write[0].W_write_mlp_in_global),
            id(model.gdh_write[0].W_write_mlp_out_global),
        }

    found_lrs = set()
    for group in opt.param_groups:
        ids = {id(p) for p in group["params"]}
        if ids & global_targets:
            found_lrs.add(group["lr"])

    assert found_lrs == {matrix_lr}

    # A local GDH matrix should remain on the base matrix LR group.
    local_id = id(model.gdh_write[0].W_k_write)
    local_lrs = {group["lr"] for group in opt.param_groups if any(id(p) == local_id for p in group["params"])}
    assert local_lrs == {matrix_lr}
