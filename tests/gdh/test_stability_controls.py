import torch

from nanochat.double_helix import GDHWriteCore
from nanochat.gpt import GPT, GPTConfig


def test_write_value_tanh_bounds_delta_range():
    torch.manual_seed(0)
    d, r, h = 8, 4, 2
    core = GDHWriteCore(d=d, r=r, use_write_brain=False)

    with torch.no_grad():
        core.W_k_write.zero_()
        core.W_q_slots_global.zero_()
        core.W_o_write.copy_(torch.eye(d))
        core.W_v_write.fill_(1000.0)

    local_out = torch.randn(2, 3, d)
    sidecar_prev = torch.zeros(2, 3, r, d)

    delta = core.forward_sequence_delta(local_out, sidecar_prev, n_write_heads=h, eps=1e-6)

    # With tanh-throttled write values and convex routing, each output component stays bounded.
    assert delta.abs().max().item() <= 1.0001


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
        gdh_use_read_gate=False,
        gdh_use_write_gate=True,
        gdh_use_ema_scan=True,
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
        gdh_use_read_gate=False,
        gdh_use_write_gate=True,
        gdh_use_ema_scan=True,
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


def test_gdh_global_params_use_lower_lr_group():
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
        gdh_use_read_gate=False,
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

    assert found_lrs == {matrix_lr * 0.25}

    # A local GDH matrix should remain on the base matrix LR group.
    local_id = id(model.gdh_write[0].W_k_write)
    local_lrs = {group["lr"] for group in opt.param_groups if any(id(p) == local_id for p in group["params"])}
    assert local_lrs == {matrix_lr}
