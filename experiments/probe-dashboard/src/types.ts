export type ProbeHist = {
  bins?: number[]
  values?: number[]
  stats?: Record<string, number>
}

export type ProbeHistoryPoint = {
  step: number
  beta?: number
  run_label?: string
  lr?: number
  eval_total?: number
  eval_ce?: number
  eval_usage_loss?: number
  eval_acc_top1?: number
  eval_mrr?: number
  eval_n_answers?: number
  eval_acc?: number
  train_total?: number
  train_ce?: number
  train_usage_loss?: number
  train_acc_top1?: number
  train_mrr?: number
  slot_cos_l_last?: number | null
  slot_cos_layers?: number[]
  slot_cos_max_layers?: number[]
  slot_usage_effective_slots_layers?: number[]
  slot_usage_max_share_layers?: number[]
  slot_participation_ratio_layers?: number[]
  slot_norm_mean_layers?: number[]
  slot_norm_cv_layers?: number[]
  sidecar_norm_trace_layers?: number[][][]
  sidecar_last_layers?: number[][][]
  out_state_hist_layers?: ProbeHist[]
  out_state_hist?: ProbeHist | null
  out_state_stats_layers?: Record<string, number>[]
  out_state_stats?: Record<string, number> | null
  [key: string]: unknown
}

export type ProbeRun = {
  beta?: number
  run_label?: string
  first?: ProbeHistoryPoint
  last?: ProbeHistoryPoint
  history?: ProbeHistoryPoint[]
}

export type ProbePayload = {
  config?: Record<string, unknown>
  beta?: number
  betas?: number[]
  run_label?: string
  history?: ProbeHistoryPoint[]
  last?: ProbeHistoryPoint
  summary?: Array<Record<string, unknown>>
  runs?: ProbeRun[]
  status?: string
  error?: string
}
