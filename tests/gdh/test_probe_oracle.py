import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

from tests.gdh.probe_oracle import (
    forward_gdh_probe_oracle,
    scan_accumulate_oracle,
    usage_balance_loss_oracle,
    write_delta_and_alpha_oracle,
)


def _load_probe_module():
    repo_root = Path(__file__).resolve().parents[2]
    probe_path = repo_root / "experiments" / "archive" / "scan-probe" / "mqar_scan_beta_probe.py"
    spec = importlib.util.spec_from_file_location("mqar_scan_beta_probe_mod", probe_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe_mod():
    return _load_probe_module()


def _make_args(**overrides):
    base = dict(
        # core
        arch="gdh",
        seed=123,
        # model
        sequence_len=24,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=32,
        gdh_slots=4,
        gdh_write_heads=2,
        # probe knobs
        route_topk=2,
        usage_balance_lambda=0.01,
        swa_window=0,
        # data
        n_pairs=3,
        n_queries=2,
        gap_min=4,
        gap_max=8,
        key_vocab=32,
        value_vocab=32,
        query_offset=64,
        filler_offset=96,
        filler_vocab=32,
        # eval
        eval_topk=5,
        batch_size=2,
        eval_batch_size=2,
        lr=3e-4,
        steps=10,
        log_every=5,
        betas="1.0",
        enforce_capacity_stress=False,
        device="cpu",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_probe_scan_accumulate_matches_probe_impl(probe_mod):
    torch.manual_seed(0)
    delta = torch.randn(2, 7, 4, 5)

    for beta in [1.0, 0.99, 0.8, 0.2, 0.0, -0.5]:
        got = scan_accumulate_oracle(delta, scan_beta=beta)
        exp = probe_mod._scan_accumulate(delta, scan_beta=beta)
        assert torch.allclose(got, exp, atol=1e-6, rtol=1e-6)


def test_probe_write_topk_properties_and_dense_fallbacks(probe_mod):
    args = _make_args(route_topk=2)
    model = probe_mod._build_model(args, device="cpu")

    torch.manual_seed(1)
    local_out = torch.randn(2, 5, args.n_embd)
    write_core = model.gdh_write[0]
    gate_proj = model.g_write_projs[0]

    # Sparse top-k case
    _, alpha, alpha_soft, _ = write_delta_and_alpha_oracle(
        write_core,
        local_out,
        n_write_heads=args.gdh_write_heads,
        route_topk=2,
        eps=1e-6,
        gate_proj=gate_proj,
    )

    # Exactly K active slots per (B,N,h) and renormalized to sum 1.
    nnz = (alpha > 0).sum(dim=-1)
    assert torch.equal(nnz, torch.full_like(nnz, 2))
    assert torch.allclose(alpha.sum(dim=-1), torch.ones_like(alpha.sum(dim=-1)), atol=1e-6, rtol=1e-6)

    # Dense fallbacks: K<=0 and K>=R should not hard-mask.
    _, alpha_k0, alpha_soft_k0, _ = write_delta_and_alpha_oracle(
        write_core,
        local_out,
        n_write_heads=args.gdh_write_heads,
        route_topk=0,
        eps=1e-6,
        gate_proj=gate_proj,
    )
    assert torch.allclose(alpha_k0, alpha_soft_k0, atol=1e-6, rtol=1e-6)

    _, alpha_kbig, alpha_soft_kbig, _ = write_delta_and_alpha_oracle(
        write_core,
        local_out,
        n_write_heads=args.gdh_write_heads,
        route_topk=args.gdh_slots,
        eps=1e-6,
        gate_proj=gate_proj,
    )
    assert torch.allclose(alpha_kbig, alpha_soft_kbig, atol=1e-6, rtol=1e-6)


def test_probe_usage_balance_gate_weighting_formula():
    # [B,N,h,R] with one head and 2 tokens.
    alpha_soft = torch.tensor(
        [[
            [[0.9, 0.1, 0.0, 0.0]],  # token 0
            [[0.1, 0.9, 0.0, 0.0]],  # token 1
        ]],
        dtype=torch.float32,
    )
    # Make token 0 effectively ignored and token 1 counted.
    g = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)

    got = usage_balance_loss_oracle(alpha_soft, g, n_write_heads=1)

    weighted_sum = (alpha_soft * g).sum(dim=(0, 1, 2))
    usage = weighted_sum / (g.sum() + 1e-9)
    target = torch.full_like(usage, 0.25)
    exp = (usage - target).pow(2).mean()

    assert torch.allclose(got, exp, atol=1e-8, rtol=1e-8)


def test_probe_build_model_injects_g_write_with_negative_bias(probe_mod):
    args = _make_args()
    model = probe_mod._build_model(args, device="cpu")

    assert hasattr(model, "g_write_projs")
    assert len(model.g_write_projs) == args.n_layer

    for proj in model.g_write_projs:
        assert proj.weight.shape == (1, args.n_embd)
        assert torch.allclose(proj.bias.detach(), torch.full_like(proj.bias.detach(), -2.0), atol=0.0, rtol=0.0)


def _assert_probe_forward_close(out_oracle, out_probe, *, eval_topk: int, vocab_size: int):
    total_o, ce_o, usage_o, metrics_o, stats_o = out_oracle
    total_p, ce_p, usage_p, metrics_p, stats_p = out_probe

    assert torch.allclose(total_o, total_p, atol=2e-5, rtol=1e-5)
    assert torch.allclose(ce_o, ce_p, atol=2e-5, rtol=1e-5)
    assert torch.allclose(usage_o, usage_p, atol=2e-5, rtol=1e-5)

    topk_key = f"acc_top{max(1, min(eval_topk, vocab_size))}"
    for k in ["acc_top1", topk_key, "mrr", "n_answers"]:
        assert abs(float(metrics_o[k]) - float(metrics_p[k])) < 1e-6

    assert stats_o is not None and stats_p is not None
    assert len(stats_o["slot_cos_mean_last_per_layer"]) == len(stats_p["slot_cos_mean_last_per_layer"])
    for a, b in zip(stats_o["slot_cos_mean_last_per_layer"], stats_p["slot_cos_mean_last_per_layer"]):
        assert abs(float(a) - float(b)) < 1e-6


def test_probe_oracle_forward_matches_probe_forward_dense_beta1(probe_mod):
    args = _make_args(route_topk=0)
    model = probe_mod._build_model(args, device="cpu")

    torch.manual_seed(args.seed + 1)
    idx, tgt = probe_mod.make_mqar_batch(args, batch_size=args.eval_batch_size, device="cpu")

    with torch.no_grad():
        out_probe = probe_mod._forward_gdh(
            model,
            idx,
            tgt,
            scan_beta=1.0,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )
        out_oracle = forward_gdh_probe_oracle(
            model,
            idx,
            tgt,
            scan_beta=1.0,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )

    _assert_probe_forward_close(out_oracle, out_probe, eval_topk=args.eval_topk, vocab_size=args.vocab_size)


def test_probe_oracle_forward_matches_probe_forward_sparse_leaky(probe_mod):
    args = _make_args(route_topk=2)
    model = probe_mod._build_model(args, device="cpu")

    torch.manual_seed(args.seed + 2)
    idx, tgt = probe_mod.make_mqar_batch(args, batch_size=args.eval_batch_size, device="cpu")

    with torch.no_grad():
        out_probe = probe_mod._forward_gdh(
            model,
            idx,
            tgt,
            scan_beta=0.9,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )
        out_oracle = forward_gdh_probe_oracle(
            model,
            idx,
            tgt,
            scan_beta=0.9,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )

    _assert_probe_forward_close(out_oracle, out_probe, eval_topk=args.eval_topk, vocab_size=args.vocab_size)


def test_probe_oracle_forward_matches_probe_forward_with_read_gate(probe_mod):
    args = _make_args(route_topk=2)
    model = probe_mod._build_model(args, device="cpu")

    torch.manual_seed(args.seed + 3)
    idx, tgt = probe_mod.make_mqar_batch(args, batch_size=args.eval_batch_size, device="cpu")

    with torch.no_grad():
        out_probe = probe_mod._forward_gdh(
            model,
            idx,
            tgt,
            scan_beta=1.0,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )
        out_oracle = forward_gdh_probe_oracle(
            model,
            idx,
            tgt,
            scan_beta=1.0,
            route_topk=args.route_topk,
            usage_balance_lambda=args.usage_balance_lambda,
            eval_topk=args.eval_topk,
            collect_sidecar=True,
        )

    _assert_probe_forward_close(out_oracle, out_probe, eval_topk=args.eval_topk, vocab_size=args.vocab_size)
