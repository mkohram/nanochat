#!/usr/bin/env python3
"""Tiny standalone live dashboard for probe runs.

Usage:
  .venv/bin/python scripts/probe_live_dashboard.py --live-json experiments/out/probe_live.json --port 8787
Then open: http://localhost:8787
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def make_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Probe Live Dashboard</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    .section-title { margin: 16px 0 8px 0; padding: 6px 10px; border-left: 4px solid #2563eb; background: #f8fafc; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; min-height: 300px; }
    .card h4 { margin: 2px 0 8px 0; font-size: 14px; }
    .card canvas { width: 100% !important; height: 250px !important; display: block; }
    .meta { margin-bottom: 8px; font-family: ui-monospace, monospace; }
    .cfg { margin-bottom: 12px; font-family: ui-monospace, monospace; white-space: pre-wrap; border: 1px solid #ddd; border-radius: 8px; padding: 8px; }
  </style>
</head>
<body>
  <h2>Probe Live Dashboard</h2>
  <div class=\"meta\" id=\"meta\">loading...</div>
  <div class=\"cfg\" id=\"cfg\">config loading...</div>

  <h3 class=\"section-title\">Learning</h3>
  <div class=\"grid\">
    <div class=\"card\"><h4>Eval Acc Top-1</h4><canvas id=\"acc\"></canvas></div>
    <div class=\"card\"><h4>Eval MRR</h4><canvas id=\"mrr\"></canvas></div>
    <div class=\"card\"><h4>Train vs Eval CE (overlay)</h4><canvas id=\"ce\"></canvas></div>
    <div class=\"card\"><h4>Latest Out-state Histogram Overlay (Per Layer)</h4><canvas id=\"hist\"></canvas></div>
  </div>

  <h3 class=\"section-title\">Slot Collapse</h3>
  <div class=\"grid\">
    <div class=\"card\"><h4>Layer-wise Slot Cosine Trend (mean off-diag)</h4><canvas id=\"layercos\"></canvas></div>
    <div class=\"card\"><h4>Layer-wise Slot Cosine Trend (max off-diag)</h4><canvas id=\"layercosmax\"></canvas></div>
    <div class=\"card\"><h4>Effective Slots (exp entropy)</h4><canvas id=\"effslots\"></canvas></div>
    <div class=\"card\"><h4>Max Slot Share</h4><canvas id=\"maxshare\"></canvas></div>
    <div class=\"card\"><h4>Participation Ratio</h4><canvas id=\"pr\"></canvas></div>
    <div class=\"card\"><h4>Slot Norm Mean</h4><canvas id=\"normmean\"></canvas></div>
    <div class=\"card\"><h4>Slot Norm CV</h4><canvas id=\"normcv\"></canvas></div>
  </div>
<script>
const baseOpts = {
  animation: false,
  responsive: true,
  maintainAspectRatio: false,
  resizeDelay: 100,
  plugins: { title: { display: false, text: '' } },
  scales: { x: { title: {display:true, text:'step'} } },
};

const layerColors = ['#0ea5e9','#22c55e','#f97316','#a855f7','#ef4444','#14b8a6','#eab308','#64748b'];

const mk = (id, label, color='#2563eb') => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [{label, data: [], borderColor: color, pointRadius: 2}] },
  options: baseOpts,
});

const mkLayer = (id) => new Chart(document.getElementById(id), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: baseOpts,
});

const charts = {
  acc: mk('acc', 'eval_acc_top1', '#2563eb'),
  mrr: mk('mrr', 'eval_mrr', '#7c3aed'),
  ce: new Chart(document.getElementById('ce'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'eval_ce', data: [], borderColor: '#dc2626', pointRadius: 2 },
      { label: 'train_ce', data: [], borderColor: '#0ea5e9', pointRadius: 2 },
    ]},
    options: baseOpts,
  }),
  layercos: mkLayer('layercos'),
  layercosmax: mkLayer('layercosmax'),
  effslots: mkLayer('effslots'),
  maxshare: mkLayer('maxshare'),
  pr: mkLayer('pr'),
  normmean: mkLayer('normmean'),
  normcv: mkLayer('normcv'),
  hist: mkLayer('hist'),
};

function nz(v, d) { return (v === null || v === undefined) ? d : v; }

function setSeries(chart, labels, data) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = data;
  chart.update();
}

function setLayerSeries(chart, steps, rows, key) {
  const last = rows.length > 0 ? rows[rows.length - 1] : {};
  const lastVals = (last && Array.isArray(last[key])) ? last[key] : [];
  if (!lastVals.length) {
    chart.data.labels = [];
    chart.data.datasets = [];
    chart.update();
    return;
  }

  const n = lastVals.length;
  chart.data.labels = steps;
  chart.data.datasets = Array.from({length:n}, (_,i)=>({
    label:`layer ${i}`,
    data: rows.map(r => ((r[key] || [])[i] ?? null)),
    borderColor: layerColors[i % layerColors.length],
    pointRadius: 1,
  }));
  chart.update();
}

function lastLayerVal(last, key) {
  const arr = (last && Array.isArray(last[key])) ? last[key] : [];
  return arr.length ? arr[arr.length - 1] : null;
}

async function refresh() {
  try {
    const res = await fetch('/data?ts=' + Date.now(), { cache: 'no-store' });
    const d = await res.json();
    const h = d.history || [];
    const steps = h.map(x => x.step);

    setSeries(charts.acc, steps, h.map(x => nz(x.eval_acc_top1, null)));
    setSeries(charts.mrr, steps, h.map(x => nz(x.eval_mrr, null)));

    charts.ce.data.labels = steps;
    charts.ce.data.datasets[0].data = h.map(x => nz(x.eval_ce, null));
    charts.ce.data.datasets[1].data = h.map(x => nz(x.train_ce, null));
    charts.ce.update();

    setLayerSeries(charts.layercos, steps, h, 'slot_cos_layers');
    setLayerSeries(charts.layercosmax, steps, h, 'slot_cos_max_layers');
    setLayerSeries(charts.effslots, steps, h, 'slot_usage_effective_slots_layers');
    setLayerSeries(charts.maxshare, steps, h, 'slot_usage_max_share_layers');
    setLayerSeries(charts.pr, steps, h, 'slot_participation_ratio_layers');
    setLayerSeries(charts.normmean, steps, h, 'slot_norm_mean_layers');
    setLayerSeries(charts.normcv, steps, h, 'slot_norm_cv_layers');

    const meta = document.getElementById('meta');
    const last = d.last || {};
    const st = last.out_state_stats || {};
    const effLast = lastLayerVal(last, 'slot_usage_effective_slots_layers');
    const maxShareLast = lastLayerVal(last, 'slot_usage_max_share_layers');
    const prLast = lastLayerVal(last, 'slot_participation_ratio_layers');

    const statsTxt = (st && Object.keys(st).length > 0)
      ? ` | out[min=${Number(st.min).toFixed(3)} p01=${Number(st.p01).toFixed(3)} p50=${Number(st.p50).toFixed(3)} p99=${Number(st.p99).toFixed(3)} max=${Number(st.max).toFixed(3)} std=${Number(st.std).toFixed(3)}]`
      : '';
    const collapseTxt = (effLast !== null)
      ? ` | collapse[Llast eff=${Number(effLast).toFixed(2)} max_share=${Number(maxShareLast).toFixed(3)} pr=${Number(prLast).toFixed(2)}]`
      : '';
    const nowTxt = new Date().toLocaleTimeString();
    meta.textContent = `beta=${nz(d.beta,'na')} | points=${h.length} | step=${nz(last.step,'na')} | acc=${nz(last.eval_acc_top1,'na')} | refreshed=${nowTxt}${collapseTxt}${statsTxt}`;

    const cfg = d.config || {};
    const keys = [
      'arch','betas','steps','log_every','seed','sequence_len','vocab_size',
      'n_layer','n_head','n_embd','gdh_slots','gdh_write_heads','gdh_use_write_brain',
      'route_topk','usage_balance_lambda','swa_window','n_pairs','n_queries','gap_min','gap_max',
      'batch_size','eval_batch_size','lr','lr_decay_iters','min_lr','device'
    ];
    const cfgLines = keys.filter(k => cfg[k] !== undefined).map(k => `${k}=${cfg[k]}`);
    document.getElementById('cfg').textContent = cfgLines.join(' | ');

    // latest out-state hist overlay per layer
    const hists = (d.last || {}).out_state_hist_layers || [];
    if (hists.length > 0) {
      charts.hist.data.labels = hists[0].bins.slice(0, -1).map((v, i) => (0.5*(hists[0].bins[i]+hists[0].bins[i+1])).toFixed(2));
      charts.hist.data.datasets = hists.map((hh, i) => {
        const vals = hh.values || [];
        const bins = hh.bins || [];
        let dens = vals;
        if (bins.length === vals.length + 1) {
          dens = vals.map((v, j) => {
            const w = bins[j+1]-bins[j];
            return w > 0 ? v / w : 0;
          });
          const s = dens.reduce((a,b)=>a+b,0);
          if (s > 0) dens = dens.map(x => x/s);
        }
        return { label:`layer ${i}`, data: dens, borderColor: layerColors[i % layerColors.length], pointRadius: 0 };
      });
      charts.hist.update();
    }
  } catch (e) {
    document.getElementById('meta').textContent = 'waiting for live json...';
  }
}

window.addEventListener('resize', () => {
  Object.values(charts).forEach(ch => { try { ch.resize(); } catch(e) {} });
});

setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""


def make_handler(live_json_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html = make_html().encode("utf-8")
                self._send(200, html, "text/html; charset=utf-8")
                return
            if self.path.startswith("/data"):
                if live_json_path.exists():
                    try:
                        data = json.loads(live_json_path.read_text())
                    except Exception:
                        data = {"history": [], "error": "failed to parse live json"}
                else:
                    data = {"history": [], "status": "waiting"}
                self._send(200, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")
                return
            self._send(404, b"not found")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-json", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    live_json_path = Path(args.live_json)
    server = HTTPServer((args.host, args.port), make_handler(live_json_path))
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Watching: {live_json_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
