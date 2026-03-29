import argparse
import importlib.util
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[2]
    probe_path = repo_root / "experiments" / "archive" / "scan-probe" / "mqar_scan_beta_probe.py"
    spec = importlib.util.spec_from_file_location("mqar_scan_beta_probe_mod", probe_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_args(**overrides):
    base = dict(
        arch="gdh",
        seed=123,
        sequence_len=64,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_embd=64,
        gdh_slots=8,
        gdh_write_heads=4,
        route_topk=2,
        usage_balance_lambda=0.01,
        swa_window=0,
        n_pairs=4,
        n_queries=4,
        gap_min=8,
        gap_max=16,
        key_vocab=32,
        value_vocab=32,
        query_offset=64,
        filler_offset=96,
        filler_vocab=32,
        batch_size=4,
        eval_batch_size=4,
        eval_topk=5,
        lr=3e-4,
        steps=10,
        log_every=5,
        betas="1.0",
        enforce_capacity_stress=False,
        device="cpu",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_mainline_from_probe_model(probe_model, *, route_topk: int, scan_beta: float, usage_balance_lambda: float):
    cfg = GPTConfig(
        sequence_len=probe_model.config.sequence_len,
        vocab_size=probe_model.config.vocab_size,
        n_layer=probe_model.config.n_layer,
        n_head=probe_model.config.n_head,
        n_kv_head=probe_model.config.n_kv_head,
        n_embd=probe_model.config.n_embd,
        window_pattern=probe_model.config.window_pattern,
        arch="gdh",
        gdh_slots=probe_model.config.gdh_slots,
        gdh_write_heads=probe_model.config.gdh_write_heads,
        gdh_use_write_brain=True,
        gdh_write_brain_hidden_mult=4,
        gdh_route_topk=route_topk,
        gdh_scan_beta=scan_beta,
        gdh_usage_balance_lambda=usage_balance_lambda,
        gdh_use_write_gate=True,
        gdh_write_gate_bias=-2.0,
    )
    model = GPT(cfg)
    model.init_weights()

    # Copy shared weights.
    model.load_state_dict(probe_model.state_dict(), strict=False)

    # Map probe-injected gates -> mainline gates.
    with torch.no_grad():
        for i in range(cfg.n_layer):
            model.gdh_write_gate[i].weight.copy_(probe_model.g_write_projs[i].weight)
            model.gdh_write_gate[i].bias.copy_(probe_model.g_write_projs[i].bias)

    model.eval()
    return model


def test_mainline_matches_probe_dense_beta1_total_loss():
    probe = _load_probe_module()
    args = _make_args(route_topk=0)

    probe_model = probe._build_model(args, device="cpu")
    probe_model.eval()

    main_model = _build_mainline_from_probe_model(
        probe_model,
        route_topk=0,
        scan_beta=1.0,
        usage_balance_lambda=args.usage_balance_lambda,
    )

    torch.manual_seed(args.seed + 7)
    idx, tgt = probe.make_mqar_batch(args, batch_size=args.eval_batch_size, device="cpu")

    with torch.no_grad():
        total_probe, _ce_probe, _usage_probe, _metrics, _stats = probe._forward_gdh(
            probe_model,
            idx,
            tgt,
            scan_beta=1.0,
            route_topk=0,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )
        total_main = main_model(idx, targets=tgt)

    # Numerical parity: allow tiny drift from implementation-order differences
    # (segmented scan stability guards / op ordering), while enforcing close match.
    assert torch.allclose(total_main, total_probe, atol=1e-3, rtol=1e-4)


def test_mainline_matches_probe_sparse_leaky_total_loss():
    probe = _load_probe_module()
    args = _make_args(route_topk=2)

    probe_model = probe._build_model(args, device="cpu")
    probe_model.eval()

    main_model = _build_mainline_from_probe_model(
        probe_model,
        route_topk=2,
        scan_beta=0.9,
        usage_balance_lambda=args.usage_balance_lambda,
    )

    torch.manual_seed(args.seed + 8)
    idx, tgt = probe.make_mqar_batch(args, batch_size=args.eval_batch_size, device="cpu")

    with torch.no_grad():
        total_probe, _ce_probe, _usage_probe, _metrics, _stats = probe._forward_gdh(
            probe_model,
            idx,
            tgt,
            scan_beta=0.9,
            route_topk=2,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )
        total_main = main_model(idx, targets=tgt)

    assert torch.allclose(total_main, total_probe, atol=1e-3, rtol=1e-4)
