#!/usr/bin/env python3
"""Generate comparison plots/report for two W&B runs.

Example:
  .venv/bin/python -m scripts.compare_runs_report \
    --baseline mkohram-none/nanochat/sct67cz9 \
    --candidate mkohram-none/nanochat/fnouhifk \
    --outdir reports/run_compare_latest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import wandb


def plot_layerwise_candidate_metric(run, metric_template: str, n_layers: int, out_png: Path, title: str, step_key: str = "step"):
    """Plot candidate-only metric with all layers overlaid.

    metric_template example: "gdh/layer_{i}/write_gate_mean"
    """
    plotted = 0
    plt.figure(figsize=(10, 5), dpi=160)
    max_step = 0
    for i in range(n_layers):
        key = metric_template.format(i=i)
        xs, ys = fetch_metric(run, key, step_key=step_key)
        if not xs:
            continue
        plotted += 1
        max_step = max(max_step, max(xs))
        plt.plot(xs, ys, lw=1.2, label=f"layer {i}")
    if plotted == 0:
        plt.close()
        return False
    plt.xlabel("train step")
    plt.ylabel(metric_template.replace("{i}", "<layer>"))
    plt.title(f"{title} (candidate, up to step {max_step})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return True


def fetch_metric(run, metric: str, step_key: str = "step") -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for row in run.scan_history(keys=[step_key, metric]):
        s = row.get(step_key)
        y = row.get(metric)
        if s is None or y is None:
            continue
        xs.append(int(s))
        ys.append(float(y))
    return xs, ys


def compare_metric(
    baseline_map: Dict[int, float],
    candidate_map: Dict[int, float],
    out_png_overlay: Path,
    out_png_delta: Path,
    metric: str,
    max_step: int,
) -> Dict[str, float]:
    common = sorted(set(baseline_map).intersection(candidate_map))
    if not common:
        return {"common_points": 0, "max_step": max_step}

    b = [baseline_map[s] for s in common]
    c = [candidate_map[s] for s in common]
    d = [cv - bv for cv, bv in zip(c, b)]

    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(common, b, label="baseline", lw=1.2)
    plt.plot(common, c, label="candidate", lw=1.2)
    plt.xlabel("train step")
    plt.ylabel(metric)
    plt.title(f"{metric} overlay (up to step {max_step})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_overlay)
    plt.close()

    plt.figure(figsize=(10, 5), dpi=160)
    plt.plot(common, d, color="purple", lw=1.2, label="delta = candidate - baseline")
    plt.axhline(0, color="black", lw=1, alpha=0.7)
    plt.xlabel("train step")
    plt.ylabel(f"delta({metric})")
    plt.title(f"{metric} delta (up to step {max_step})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_delta)
    plt.close()

    return {
        "common_points": len(common),
        "max_step": max_step,
        "delta_last": d[-1],
        "delta_mean": sum(d) / len(d),
        "baseline_last": b[-1],
        "candidate_last": c[-1],
    }


def build_out_state_hist_overlay(run, out_png: Path, n_layers: int):
    plt.figure(figsize=(10, 5), dpi=160)
    plotted = 0
    for i in range(n_layers):
        key = f"gdh/layer_{i}/out_state_hist"
        latest = None
        for row in run.scan_history(keys=["step", key]):
            h = row.get(key)
            if isinstance(h, dict) and "bins" in h and "values" in h:
                latest = h
        if latest is None:
            continue
        bins = np.array(latest["bins"], dtype=float)
        vals = np.array(latest["values"], dtype=float)
        if bins.size == vals.size + 1:
            centers = 0.5 * (bins[:-1] + bins[1:])
            widths = np.diff(bins)
            dens = vals / np.maximum(vals.sum() * widths, 1e-12)
        else:
            centers = np.arange(vals.size)
            dens = vals / np.maximum(vals.sum(), 1e-12)
        plt.plot(centers, dens, lw=1.2, label=f"layer {i}")
        plotted += 1
    if plotted == 0:
        plt.close()
        return False
    plt.title("Latest GDH out_state histogram overlay (candidate)")
    plt.xlabel("out_state value")
    plt.ylabel("density")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    return True


def build_onepager(outdir: Path, title: str):
    candidates = [
        ("Overall Loss Overlay", outdir / "train_loss_overlay.png"),
        ("Overall Loss Delta", outdir / "train_loss_delta.png"),
        ("Chunk 9 Overlay", outdir / "train_chunk_loss_avg_9_overlay.png"),
        ("Chunk 9 Delta", outdir / "train_chunk_loss_avg_9_delta.png"),
        ("Gap vs 0 Overlay", outdir / "train_chunk_loss_gap_vs_0_overlay.png"),
        ("Gap vs 0 Delta", outdir / "train_chunk_loss_gap_vs_0_delta.png"),
        ("Out-State Hist Overlay", outdir / "gdh_out_state_hist_overlay_latest.png"),
        ("Layer-wise Gate Mean", outdir / "gdh_layerwise_write_gate_mean_candidate.png"),
        ("Layer-wise Read Mute Mean", outdir / "gdh_layerwise_read_mute_mean_candidate.png"),
    ]
    panels = [(t, p) for (t, p) in candidates if p.exists()]
    if not panels:
        return None
    n = len(panels)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 6 * rows), dpi=180)
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    fig.suptitle(title, fontsize=18, y=0.995)
    for idx, ax in enumerate(axes):
        if idx < n:
            panel_title, path = panels[idx]
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(panel_title, fontsize=12)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = outdir / "comparison_report_onepager.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate baseline vs candidate comparison report from W&B")
    ap.add_argument("--baseline", required=True, help="W&B run path, e.g. entity/project/run_id")
    ap.add_argument("--candidate", required=True, help="W&B run path, e.g. entity/project/run_id")
    ap.add_argument("--metrics", default="train/loss,train/chunk_loss_avg_9,train/chunk_loss_gap_vs_0")
    ap.add_argument("--step-key", default="step")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    rb = api.run(args.baseline)
    rc = api.run(args.candidate)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if isinstance(rc.config, dict) and "n_layer" in rc.config:
        n_layers = int(rc.config.get("n_layer", 0))
    else:
        # Fallback: infer from available layerwise summary keys.
        layer_ids = []
        for k in rc.summary.keys():
            if k.startswith("gdh/layer_"):
                try:
                    layer_ids.append(int(k.split("/")[1].split("_")[1]))
                except Exception:
                    pass
        n_layers = (max(layer_ids) + 1) if layer_ids else 0

    summary_lines = [
        f"# Run Comparison Report",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Candidate: `{args.candidate}`",
        "",
    ]

    for metric in metrics:
        xb, yb = fetch_metric(rb, metric, step_key=args.step_key)
        xc, yc = fetch_metric(rc, metric, step_key=args.step_key)
        if not xc:
            summary_lines.append(f"## {metric}\nNo candidate points found.\n")
            continue

        max_step = max(xc)
        bmap = {x: y for x, y in zip(xb, yb) if x <= max_step}
        cmap = {x: y for x, y in zip(xc, yc)}

        overlay = outdir / f"{metric.replace('/', '_')}_overlay.png"
        delta = outdir / f"{metric.replace('/', '_')}_delta.png"

        stats = compare_metric(bmap, cmap, overlay, delta, metric, max_step)

        summary_lines.append(f"## {metric}")
        summary_lines.append(f"- overlay: `{overlay.name}`")
        summary_lines.append(f"- delta: `{delta.name}`")
        for k, v in stats.items():
            summary_lines.append(f"- {k}: {v}")
        summary_lines.append("")

    # Candidate-only layerwise gate diagnostics
    if n_layers > 0:
        # Prefer retention naming in EMA runs; fallback to write-gate naming for additive runs.
        write_gate_plot = outdir / "gdh_layerwise_write_gate_mean_candidate.png"
        plotted = plot_layerwise_candidate_metric(
            rc,
            metric_template="gdh/layer_{i}/retention_gate_mean",
            n_layers=n_layers,
            out_png=write_gate_plot,
            title="Layer-wise retention gate mean",
            step_key=args.step_key,
        )
        if not plotted:
            plotted = plot_layerwise_candidate_metric(
                rc,
                metric_template="gdh/layer_{i}/write_gate_mean",
                n_layers=n_layers,
                out_png=write_gate_plot,
                title="Layer-wise write gate mean",
                step_key=args.step_key,
            )
        if plotted:
            summary_lines.append("## candidate layerwise gate mean")
            summary_lines.append(f"- plot: `{write_gate_plot.name}`")
            summary_lines.append("")

        read_mute_plot = outdir / "gdh_layerwise_read_mute_mean_candidate.png"
        if plot_layerwise_candidate_metric(
            rc,
            metric_template="gdh/layer_{i}/read_mute_mean",
            n_layers=n_layers,
            out_png=read_mute_plot,
            title="Layer-wise read mute mean",
            step_key=args.step_key,
        ):
            summary_lines.append("## candidate layerwise read mute gate")
            summary_lines.append(f"- plot: `{read_mute_plot.name}`")
            summary_lines.append("")

        out_state_plot = outdir / "gdh_out_state_hist_overlay_latest.png"
        if build_out_state_hist_overlay(rc, out_state_plot, n_layers=n_layers):
            summary_lines.append("## candidate out-state distribution")
            summary_lines.append(f"- plot: `{out_state_plot.name}`")
            summary_lines.append("")

    onepager = build_onepager(outdir, title="Run Comparison Report (auto onepager)")
    if onepager is not None:
        summary_lines.append("## onepager")
        summary_lines.append(f"- image: `{onepager.name}`")
        summary_lines.append("")

    (outdir / "REPORT.md").write_text("\n".join(summary_lines))
    print(outdir / "REPORT.md")


if __name__ == "__main__":
    main()
