import { useEffect, useMemo, useRef, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { ProbePayload, ProbeHistoryPoint, ProbeHist } from './types'

declare global {
  interface Window {
    __PROBE_DASHBOARD_DEFAULT_URL__?: string
  }
}

const POLL_MS = 2000
const LAYER_COLORS = ['#38bdf8', '#22c55e', '#f97316', '#a855f7', '#ef4444', '#14b8a6', '#eab308', '#94a3b8']

function fmt(v: unknown, digits = 4): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

function compactConfig(cfg?: Record<string, unknown>): string {
  if (!cfg) return 'No config loaded.'
  const keys = [
    'arch','data_source','betas','steps','log_every','seed','sequence_len','vocab_size',
    'n_layer','n_head','n_embd','gdh_slots','gdh_write_heads',
    'gdh_use_write_brain','gdh_write_brain_hidden_mult','read_mute_gate',
    'route_topk','write_routing','state_mixer','write_cooloff_lambda','write_cooloff_rho',
    'future_summary_horizon','future_summary_lambda','future_summary_hidden_mult',
    'swa_window','n_pairs','n_queries',
    'gap_min','gap_max','batch_size','grad_accum_steps','eval_batch_size','lr','lr_decay_iters','min_lr','device',
  ]
  return keys.filter((k) => cfg[k] !== undefined).map((k) => `${k}=${String(cfg[k])}`).join(' | ')
}

function flattenPayload(payload: ProbePayload): { history: ProbeHistoryPoint[]; last?: ProbeHistoryPoint; runLabel?: string } {
  if (payload.history) {
    return { history: payload.history, last: payload.last ?? payload.history[payload.history.length - 1], runLabel: payload.run_label ?? payload.last?.run_label }
  }
  const run = payload.runs?.[0]
  const history = run?.history ?? []
  return { history, last: run?.last ?? history[history.length - 1], runLabel: run?.run_label ?? run?.last?.run_label }
}

function paddedDomain(values: Array<number | null | undefined>, padFrac = 0.08): [number, number] | undefined {
  const nums = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  if (nums.length === 0) return undefined
  let lo = Math.min(...nums)
  let hi = Math.max(...nums)
  if (lo === hi) {
    const pad = Math.max(0.1, Math.abs(lo) * padFrac)
    return [lo - pad, hi + pad]
  }
  const pad = (hi - lo) * padFrac
  return [lo - pad, hi + pad]
}

function paddedPositiveDomain(values: Array<number | null | undefined>, padFrac = 0.08, minLo = 1): [number, number] | undefined {
  const domain = paddedDomain(values, padFrac)
  if (!domain) return undefined
  return [Math.max(minLo, domain[0]), Math.max(minLo, domain[1])]
}

function carryForward(values: Array<number | null | undefined>): Array<number | null> {
  let last: number | null = null
  return values.map((v) => {
    if (typeof v === 'number' && Number.isFinite(v)) {
      last = v
      return v
    }
    return last
  })
}

function lastNumber(values: unknown): number | null {
  if (!Array.isArray(values) || values.length === 0) return null
  const v = values[values.length - 1]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

function vocabRandomTop1(vocabSize: unknown): number | null {
  return typeof vocabSize === 'number' && Number.isFinite(vocabSize) && vocabSize > 0 ? 1 / vocabSize : null
}

function vocabRandomMRR(vocabSize: unknown): number | null {
  if (typeof vocabSize !== 'number' || !Number.isFinite(vocabSize) || vocabSize <= 0) return null
  let h = 0
  for (let i = 1; i <= vocabSize; i += 1) h += 1 / i
  return h / vocabSize
}

function ceToPpl(ce: unknown): number | null {
  return typeof ce === 'number' && Number.isFinite(ce) ? Math.exp(ce) : null
}

function layerSeries(history: ProbeHistoryPoint[], key: string) {
  const maxLayers = history.reduce((acc, row) => {
    const arr = row[key]
    return Array.isArray(arr) ? Math.max(acc, arr.length) : acc
  }, 0)
  return history.map((row) => {
    const out: Record<string, number | null> = { step: row.step }
    for (let i = 0; i < maxLayers; i += 1) {
      const arr = row[key]
      out[`layer_${i}`] = Array.isArray(arr) && typeof arr[i] === 'number' ? (arr[i] as number) : null
    }
    return out
  })
}

function layerSeriesDual(history: ProbeHistoryPoint[], keyA: string, keyB: string) {
  const maxLayers = history.reduce((acc, row) => {
    const arrA = row[keyA]
    const arrB = row[keyB]
    const nA = Array.isArray(arrA) ? arrA.length : 0
    const nB = Array.isArray(arrB) ? arrB.length : 0
    return Math.max(acc, nA, nB)
  }, 0)
  return history.map((row) => {
    const out: Record<string, number | null> = { step: row.step }
    for (let i = 0; i < maxLayers; i += 1) {
      const arrA = row[keyA]
      const arrB = row[keyB]
      out[`layer_${i}_max`] = Array.isArray(arrA) && typeof arrA[i] === 'number' ? (arrA[i] as number) : null
      out[`layer_${i}_p90`] = Array.isArray(arrB) && typeof arrB[i] === 'number' ? (arrB[i] as number) : null
    }
    return out
  })
}

function histOverlayData(hists?: ProbeHist[]) {
  if (!hists || hists.length === 0) return []
  const baseBins = hists[0]?.bins ?? []
  const baseValues = hists[0]?.values ?? []
  if (baseBins.length !== baseValues.length + 1) return []

  return baseValues.map((_, i) => {
    const row: Record<string, number> = {
      x: Number(((baseBins[i] + baseBins[i + 1]) * 0.5).toFixed(3)),
    }

    hists.forEach((hist, layerIdx) => {
      const bins = hist?.bins ?? []
      const values = hist?.values ?? []
      if (bins.length !== values.length + 1 || i >= values.length) {
        row[`layer_${layerIdx}`] = 0
        return
      }
      const width = bins[i + 1] - bins[i]
      row[`layer_${layerIdx}`] = width > 0 ? values[i] / width : 0
    })

    return row
  })
}

function transpose<T>(matrix: T[][]): T[][] {
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 0
  if (!rows || !cols) return []
  return Array.from({ length: cols }, (_, c) => Array.from({ length: rows }, (_, r) => matrix[r][c]))
}

function valueToFill(v: number, maxAbs: number, signed: boolean): string {
  if (!Number.isFinite(v)) return 'rgba(100,116,139,0.15)'
  if (signed) {
    const t = Math.max(-1, Math.min(1, v / Math.max(maxAbs, 1e-9)))
    if (t >= 0) {
      const a = Math.pow(t, 0.8)
      return `rgba(34,197,94,${0.10 + 0.90 * a})`
    }
    const a = Math.pow(Math.abs(t), 0.8)
    return `rgba(239,68,68,${0.10 + 0.90 * a})`
  }
  const t = Math.max(0, Math.min(1, v / Math.max(maxAbs, 1e-9)))
  const a = Math.pow(t, 0.55)
  const r = Math.round(34 + (56 - 34) * t)
  const g = Math.round(197 + (189 - 197) * t)
  const b = Math.round(94 + (248 - 94) * t)
  return `rgba(${r},${g},${b},${0.12 + 0.88 * a})`
}

function CanvasHeatmap({
  matrix,
  signed = false,
  transposeMatrix = false,
  pixel = 4,
}: {
  matrix: number[][]
  signed?: boolean
  transposeMatrix?: boolean
  pixel?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const z = useMemo(() => (transposeMatrix ? transpose(matrix) : matrix), [matrix, transposeMatrix])
  const rows = z.length
  const cols = z[0]?.length ?? 0

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !rows || !cols) return
    const dpr = Math.max(1, window.devicePixelRatio || 1)
    const width = cols * pixel
    const height = rows * pixel
    const maxAbs = Math.max(...z.flat().map((v) => Math.abs(v)), 1e-9)

    canvas.width = Math.max(1, Math.floor(width * dpr))
    canvas.height = Math.max(1, Math.floor(height * dpr))
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        ctx.fillStyle = valueToFill(z[r][c] ?? 0, maxAbs, signed)
        ctx.fillRect(c * pixel, r * pixel, pixel, pixel)
      }
    }
  }, [z, rows, cols, pixel, signed])

  if (!rows || !cols) return <div className="muted">No heatmap data yet.</div>

  return (
    <div className="heatmap-shell heatmap-shell-canvas">
      <canvas ref={canvasRef} title={`rows=${rows} cols=${cols}`} />
      <div className="heatmap-legend-wrap">
        {signed ? (
          <>
            <div className="heatmap-legend heatmap-legend-signed" />
            <div className="heatmap-legend-labels">
              <span>-max</span>
              <span>0</span>
              <span>+max</span>
            </div>
          </>
        ) : (
          <>
            <div className="heatmap-legend heatmap-legend-norm" />
            <div className="heatmap-legend-labels">
              <span>low</span>
              <span>mid</span>
              <span>high</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function StatCard({ title, value, subtitle }: { title: string; value: string; subtitle?: string }) {
  return (
    <div className="stat-card">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      {subtitle ? <div className="stat-subtitle">{subtitle}</div> : null}
    </div>
  )
}

function Panel({ title, children, actions }: { title: string; children: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </div>
      {children}
    </section>
  )
}

function ExportButton({ filename = 'chart.png' }: { filename?: string }) {
  const onClick = () => {
    const svg = document.querySelector('svg.recharts-surface') as SVGElement | null
    if (!svg) return
    const xml = new XMLSerializer().serializeToString(svg)
    const blob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename.replace(/\.png$/i, '.svg')
    a.click()
    URL.revokeObjectURL(url)
  }
  return <button onClick={onClick}>Export first chart (SVG)</button>
}

export function App() {
  const defaultUrl = window.__PROBE_DASHBOARD_DEFAULT_URL__ ?? '/probe-data/live.json'
  const [source, setSource] = useState(defaultUrl)
  const [payload, setPayload] = useState<ProbePayload | null>(null)
  const [status, setStatus] = useState('idle')
  const [live, setLive] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [heatmapLayer, setHeatmapLayer] = useState(0)
  const [yAxisIgnoreFirst, setYAxisIgnoreFirst] = useState(0)

  const load = async () => {
    setStatus('loading')
    try {
      const res = await fetch(source, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as ProbePayload
      setPayload(data)
      setError(null)
      setStatus('ready')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus('error')
    }
  }

  useEffect(() => {
    void load()
    if (!live) return
    const id = window.setInterval(() => void load(), POLL_MS)
    return () => window.clearInterval(id)
  }, [source, live])

  const flattened = useMemo(() => (payload ? flattenPayload(payload) : { history: [] as ProbeHistoryPoint[] }), [payload])
  const history = flattened.history
  const last = flattened.last
  const runLabel = flattened.runLabel ?? '—'

  const rawEvalMrr = history.map((row) => (typeof row.eval_mrr === 'number' ? row.eval_mrr : null))
  const shownEvalMrr = carryForward(rawEvalMrr)
  const learningData = history.map((row, i) => {
    const evalCe = typeof row.eval_ce === 'number' ? row.eval_ce : null
    const trainCe = typeof row.train_ce === 'number' ? row.train_ce : null
    const evalTotal = typeof row.eval_total === 'number' ? row.eval_total : null
    const trainTotal = typeof row.train_total === 'number' ? row.train_total : null
    const evalFutureSummaryLoss = typeof row.eval_future_summary_loss === 'number' ? row.eval_future_summary_loss : null
    const trainFutureSummaryLoss = typeof row.train_future_summary_loss === 'number' ? row.train_future_summary_loss : null
    return {
      step: row.step,
      wall_time_min: typeof row.wall_time_min === 'number' ? row.wall_time_min : null,
      eval_acc_top1: typeof row.eval_acc_top1 === 'number' ? row.eval_acc_top1 : null,
      eval_mrr: shownEvalMrr[i],
      eval_total: evalTotal,
      train_total: trainTotal,
      eval_ce: evalCe,
      train_ce: trainCe,
      eval_future_summary_loss: evalFutureSummaryLoss,
      train_future_summary_loss: trainFutureSummaryLoss,
      eval_gdh_off_ce: typeof row.eval_gdh_off_ce === 'number' ? row.eval_gdh_off_ce : null,
      eval_gdh_off_acc_top1: typeof row.eval_gdh_off_acc_top1 === 'number' ? row.eval_gdh_off_acc_top1 : null,
      eval_mem_delta_ce: typeof row.eval_mem_delta_ce === 'number' ? row.eval_mem_delta_ce : null,
      eval_mem_delta_acc_top1: typeof row.eval_mem_delta_acc_top1 === 'number' ? row.eval_mem_delta_acc_top1 : null,
      eval_ppl: ceToPpl(evalCe),
      train_ppl: ceToPpl(trainCe),
      full_metrics: Boolean(row.full_metrics),
    }
  })
  const randomTop1 = vocabRandomTop1(payload?.config?.vocab_size)
  const randomMrr = vocabRandomMRR(payload?.config?.vocab_size)
  const domainData = learningData.slice(Math.max(0, Math.min(yAxisIgnoreFirst, Math.max(0, learningData.length - 1))))
  const totalLossDomain = paddedDomain(
    domainData.flatMap((row) => [row.eval_total, row.train_total]),
    0.06,
  )
  const ceDomain = paddedDomain(
    domainData.flatMap((row) => [row.eval_ce, row.train_ce]),
    0.06,
  )
  const futureSummaryDomain = paddedDomain(
    domainData.flatMap((row) => [row.eval_future_summary_loss, row.train_future_summary_loss]),
    0.06,
  )
  const pplDomain = paddedPositiveDomain(
    domainData.flatMap((row) => [row.eval_ppl, row.train_ppl]),
    0.06,
    1,
  )
  const histData = histOverlayData(last?.out_state_hist_layers)
  const histLayerCount = last?.out_state_hist_layers?.length ?? 0
  const slotCosMaxP90Data = layerSeriesDual(history, 'state_slot_cos_max_layers', 'state_slot_cos_p90_layers')
  const availableHeatmapLayers = Math.max(last?.sidecar_norm_trace_layers?.length ?? 0, last?.sidecar_last_layers?.length ?? 0)
  const selectedHeatmapLayer = Math.min(heatmapLayer, Math.max(0, availableHeatmapLayers - 1))
  const sidecarNormTrace = (last?.sidecar_norm_trace_layers?.[selectedHeatmapLayer] as number[][] | undefined) ?? []
  const sidecarLast = (last?.sidecar_last_layers?.[selectedHeatmapLayer] as number[][] | undefined) ?? []

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Probe Dashboard</h1>
          <p className="muted">Viewer only: probe writes JSON, dashboard reads it.</p>
        </div>
        <div className="topbar-actions">
          <button onClick={() => void load()}>Refresh now</button>
          <label className="toggle">
            <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
            live polling
          </label>
        </div>
      </header>

      <Panel title="Data source">
        <div className="source-row">
          <input value={source} onChange={(e) => setSource(e.target.value)} spellCheck={false} />
          <button onClick={() => setSource(defaultUrl)}>Use default live JSON</button>
        </div>
        <div className="meta-row">
          <span>Status: <strong>{status}</strong></span>
          {error ? <span className="error">Error: {error}</span> : null}
        </div>
      </Panel>

      <div className="stats-grid">
        <StatCard title="Run label" value={runLabel} subtitle={history.length ? `${history.length} points` : 'no history yet'} />
        <StatCard title="Step" value={last ? String(last.step) : '—'} subtitle={payload?.beta !== undefined ? `beta=${payload.beta}` : undefined} />
        <StatCard title="Wall time" value={fmt(last?.wall_time_min, 2)} subtitle="minutes" />
        <StatCard title="Eval top1" value={fmt(last?.eval_acc_top1)} subtitle={`MRR ${fmt(last?.eval_mrr)}`} />
        <StatCard title="Eval total" value={fmt(last?.eval_total)} subtitle={`Train total ${fmt(last?.train_total)}`} />
        <StatCard title="Eval CE" value={fmt(last?.eval_ce)} subtitle={`Train CE ${fmt(last?.train_ce)}`} />
        <StatCard title="GDH-off CE" value={fmt(last?.eval_gdh_off_ce)} subtitle={`ΔCE ${fmt(last?.eval_mem_delta_ce)}`} />
        <StatCard title="GDH-off top1" value={fmt(last?.eval_gdh_off_acc_top1)} subtitle={`Δtop1 ${fmt(last?.eval_mem_delta_acc_top1)}`} />
        <StatCard title="Eval PPL" value={fmt(ceToPpl(last?.eval_ce), 2)} subtitle={`Train PPL ${fmt(ceToPpl(last?.train_ce), 2)}`} />
        <StatCard title="State slot cosine" value={fmt(last?.state_slot_cos_l_last ?? last?.slot_cos_l_last)} subtitle="last layer" />
        <StatCard title="Write load max" value={fmt(lastNumber(last?.write_load_max_share_layers))} subtitle={`Read load max ${fmt(lastNumber(last?.read_load_max_share_layers))}`} />
        <StatCard title="Out abs max" value={fmt(last?.out_state_stats?.abs_max)} subtitle={`std ${fmt(last?.out_state_stats?.std)}`} />
      </div>

      <Panel title="Config" actions={<ExportButton filename="probe-dashboard.svg" />}>
        <pre className="config-box">{compactConfig(payload?.config)}</pre>
      </Panel>

      <div className="section-title">Learning</div>
      <div className="learning-toolbar">
        <div className="axis-control-card">
          <div className="axis-control-copy">
            <div className="axis-control-title">Y-axis autoscale</div>
            <div className="axis-control-subtitle">Ignore the first N points for total / CE / future-summary / perplexity domain selection.</div>
          </div>
          <div className="axis-control-actions">
            <div className="axis-control-presets">
              {[0, 10, 50, 200, 1000].map((n) => (
                <button
                  key={n}
                  type="button"
                  className={`pill-button${yAxisIgnoreFirst === n ? ' is-active' : ''}`}
                  onClick={() => setYAxisIgnoreFirst(n)}
                >
                  {n}
                </button>
              ))}
            </div>
            <label className="axis-control-input-wrap">
              <span className="axis-control-input-label">Custom</span>
              <input
                className="axis-control-input"
                type="number"
                min={0}
                step={1}
                value={yAxisIgnoreFirst}
                onChange={(e) => setYAxisIgnoreFirst(Math.max(0, Number.parseInt(e.target.value || '0', 10) || 0))}
              />
            </label>
          </div>
        </div>
      </div>
      <div className="chart-grid two-up">
        <Panel title="Eval top-1 / MRR">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis yAxisId="left" domain={[0, 1]} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                <Tooltip formatter={(value: unknown) => fmt(value, 4)} />
                <Legend />
                {randomTop1 !== null ? (
                  <ReferenceLine
                    yAxisId="left"
                    y={randomTop1}
                    stroke="#64748b"
                    strokeDasharray="6 4"
                    ifOverflow="extendDomain"
                    label={{ value: `random top1 ${fmt(randomTop1, 4)}`, position: 'insideBottomRight', fill: '#475569', fontSize: 12 }}
                  />
                ) : null}
                {randomMrr !== null ? (
                  <ReferenceLine
                    yAxisId="right"
                    y={randomMrr}
                    stroke="#a855f7"
                    strokeDasharray="6 4"
                    ifOverflow="extendDomain"
                    label={{ value: `random MRR ${fmt(randomMrr, 4)}`, position: 'insideTopRight', fill: '#6d28d9', fontSize: 12 }}
                  />
                ) : null}
                <Line yAxisId="left" type="monotone" dataKey="eval_acc_top1" stroke="#2563eb" dot={false} connectNulls />
                <Line yAxisId="right" type="monotone" dataKey="eval_mrr" name="eval_mrr" stroke="#7c3aed" dot={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Wall time vs step">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis tickFormatter={(v) => fmt(v, 1)} width={54} />
                <Tooltip formatter={(value: unknown) => `${fmt(value, 2)} min`} />
                <Legend />
                <Line type="monotone" dataKey="wall_time_min" name="wall_time_min" stroke="#16a34a" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Train vs eval total loss">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis domain={totalLossDomain ?? ['auto', 'auto']} tickFormatter={(v) => fmt(v, 2)} width={54} allowDataOverflow />
                <Tooltip formatter={(value: unknown) => fmt(value, 3)} />
                <Legend />
                <Line type="monotone" dataKey="eval_total" name="eval_total" stroke="#be123c" dot={false} strokeWidth={2} connectNulls />
                <Line type="monotone" dataKey="train_total" name="train_total" stroke="#0f766e" dot={false} strokeWidth={2} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Train vs eval CE">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis domain={ceDomain ?? ['auto', 'auto']} tickFormatter={(v) => fmt(v, 2)} width={54} allowDataOverflow />
                <Tooltip formatter={(value: unknown) => fmt(value, 3)} />
                <Legend />
                <Line type="monotone" dataKey="eval_ce" name="eval_ce" stroke="#dc2626" dot={false} strokeWidth={2} connectNulls />
                <Line type="monotone" dataKey="train_ce" name="train_ce" stroke="#0ea5e9" dot={false} strokeWidth={2} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="GDH-on vs GDH-off eval CE / top1 deltas">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis yAxisId="left" tickFormatter={(v) => fmt(v, 2)} width={54} />
                <YAxis yAxisId="right" orientation="right" domain={[-1, 1]} tickFormatter={(v) => fmt(v, 2)} width={54} />
                <Tooltip formatter={(value: unknown) => fmt(value, 4)} />
                <Legend />
                <Line type="monotone" dataKey="eval_ce" name="eval_ce" stroke="#dc2626" dot={false} strokeWidth={2} connectNulls yAxisId="left" />
                <Line type="monotone" dataKey="eval_gdh_off_ce" name="eval_gdh_off_ce" stroke="#f59e0b" dot={false} strokeWidth={2} connectNulls yAxisId="left" />
                <Line type="monotone" dataKey="eval_mem_delta_ce" name="eval_mem_delta_ce" stroke="#16a34a" dot={false} strokeWidth={2} connectNulls yAxisId="left" />
                <Line type="monotone" dataKey="eval_mem_delta_acc_top1" name="eval_mem_delta_acc_top1" stroke="#7c3aed" dot={false} strokeWidth={2} connectNulls yAxisId="right" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Train vs eval future-summary loss">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis domain={futureSummaryDomain ?? ['auto', 'auto']} tickFormatter={(v) => fmt(v, 2)} width={54} allowDataOverflow />
                <Tooltip formatter={(value: unknown) => fmt(value, 3)} />
                <Legend />
                <Line type="monotone" dataKey="eval_future_summary_loss" name="eval_future_summary_loss" stroke="#7c3aed" dot={false} strokeWidth={2} connectNulls />
                <Line type="monotone" dataKey="train_future_summary_loss" name="train_future_summary_loss" stroke="#22c55e" dot={false} strokeWidth={2} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Train vs eval perplexity">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={learningData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis domain={pplDomain ?? [1, 'auto']} tickFormatter={(v) => fmt(v, 0)} width={64} allowDecimals={false} allowDataOverflow />
                <Tooltip formatter={(value: unknown) => fmt(value, 2)} />
                <Legend />
                <Line type="monotone" dataKey="eval_ppl" name="eval_ppl" stroke="#b91c1c" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="train_ppl" name="train_ppl" stroke="#0284c7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="section-title">State geometry</div>
      <div className="chart-grid two-up">
        <Panel title="Layer-wise state slot cosine (max and p90 off-diag)">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={slotCosMaxP90Data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis />
                <Tooltip />
                <Legend />
                {Object.keys(slotCosMaxP90Data[0] ?? {}).filter((k) => k.startsWith('layer_') && k.endsWith('_max')).map((seriesKey, idx) => {
                  const layerKey = seriesKey.replace(/_max$/, '')
                  return [
                    <Line key={seriesKey} type="monotone" dataKey={seriesKey} name={`${layerKey} max`} stroke={LAYER_COLORS[idx % LAYER_COLORS.length]} dot={false} />,
                    <Line key={`${layerKey}_p90`} type="monotone" dataKey={`${layerKey}_p90`} name={`${layerKey} p90`} stroke={LAYER_COLORS[idx % LAYER_COLORS.length]} strokeDasharray="6 4" dot={false} />,
                  ]
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        {[
          ['Layer-wise state slot cosine (mean off-diag)', 'state_slot_cos_layers'],
          ['State effective slots (norm-share exp entropy)', 'state_effective_slots_layers'],
          ['State max share (norm-share)', 'state_max_share_layers'],
          ['State participation ratio', 'state_participation_ratio_layers'],
          ['State slot norm mean', 'state_slot_norm_mean_layers'],
        ].map(([title, key]) => {
          const data = layerSeries(history, key)
          return (
            <Panel key={key} title={title}>
              <div className="chart-wrap">
                <ResponsiveContainer>
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {Object.keys(data[0] ?? {}).filter((k) => k.startsWith('layer_')).map((seriesKey, idx) => (
                      <Line key={seriesKey} type="monotone" dataKey={seriesKey} stroke={LAYER_COLORS[idx % LAYER_COLORS.length]} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          )
        })}

        <Panel title="Latest out-state histogram overlay">
          <div className="chart-wrap">
            <ResponsiveContainer>
              <LineChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" />
                <YAxis />
                <Tooltip />
                <Legend />
                {Array.from({ length: histLayerCount }, (_, idx) => (
                  <Line key={`hist_layer_${idx}`} type="monotone" dataKey={`layer_${idx}`} stroke={LAYER_COLORS[idx % LAYER_COLORS.length]} dot={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <div className="section-title">Routing and load</div>
      <div className="chart-grid two-up">
        {[
          ['Write load effective slots', 'write_load_effective_slots_layers'],
          ['Write load max share', 'write_load_max_share_layers'],
          ['Write attention effective slots', 'write_attn_effective_slots_layers'],
          ['Write attention max share', 'write_attn_max_share_layers'],
          ['Read load effective slots', 'read_load_effective_slots_layers'],
          ['Read load max share', 'read_load_max_share_layers'],
          ['Read attention effective slots', 'read_attn_effective_slots_layers'],
          ['Read attention max share', 'read_attn_max_share_layers'],
        ].map(([title, key]) => {
          const data = layerSeries(history, key)
          return (
            <Panel key={key} title={title}>
              <div className="chart-wrap">
                <ResponsiveContainer>
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="step" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {Object.keys(data[0] ?? {}).filter((k) => k.startsWith('layer_')).map((seriesKey, idx) => (
                      <Line key={seriesKey} type="monotone" dataKey={seriesKey} stroke={LAYER_COLORS[idx % LAYER_COLORS.length]} dot={false} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Panel>
          )
        })}
      </div>

      <div className="section-title">Sidecar heatmaps</div>
      <Panel title="Heatmap controls">
        <div className="meta-row">
          <label className="toggle">
            layer
            <select value={selectedHeatmapLayer} onChange={(e) => setHeatmapLayer(Number(e.target.value))}>
              {Array.from({ length: availableHeatmapLayers }, (_, idx) => <option key={idx} value={idx}>layer {idx}</option>)}
            </select>
          </label>
          <span className="muted">Using sample 0 from the fixed eval batch. “last” means token index [-1].</span>
        </div>
      </Panel>

      <div className="chart-grid two-up">
        <Panel title="Token × Slot norm heatmap">
          <CanvasHeatmap matrix={sidecarNormTrace.map((r) => r.map(Number))} transposeMatrix pixel={8} />
        </Panel>
        <Panel title="Last-token [-1] Slot × Channel heatmap">
          <CanvasHeatmap matrix={sidecarLast.map((r) => r.map(Number))} signed pixel={8} />
        </Panel>
      </div>

      <Panel title="Latest raw stats">
        <pre className="json-box">{JSON.stringify(last ?? {}, null, 2)}</pre>
      </Panel>
    </div>
  )
}
