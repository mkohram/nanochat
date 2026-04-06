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
  wall_time_s?: number
  wall_time_min?: number
  eval_total?: number
  eval_ce?: number
  eval_future_summary_loss?: number
  eval_future_summary_pred_norm?: number
  eval_acc_top1?: number
  eval_mrr?: number
  eval_n_answers?: number
  eval_acc?: number
  train_total?: number
  train_ce?: number
  train_future_summary_loss?: number
  train_future_summary_pred_norm?: number
  train_acc_top1?: number
  train_mrr?: number
  eval_gdh_off_ce?: number | null
  eval_gdh_off_acc_top1?: number | null
  eval_gdh_off_mrr?: number | null
  eval_mem_delta_ce?: number | null
  eval_mem_delta_acc_top1?: number | null
  eval_mem_delta_mrr?: number | null
  state_slot_cos_l_last?: number | null
  state_slot_cos_layers?: number[]
  state_slot_cos_max_layers?: number[]
  state_slot_cos_p90_layers?: number[]
  state_effective_slots_layers?: number[]
  state_max_share_layers?: number[]
  state_participation_ratio_layers?: number[]
  state_slot_norm_mean_layers?: number[]
  state_slot_norm_cv_layers?: number[]
  write_attn_effective_slots_layers?: number[]
  write_attn_max_share_layers?: number[]
  write_load_effective_slots_layers?: number[]
  write_load_max_share_layers?: number[]
  read_attn_effective_slots_layers?: number[]
  read_attn_max_share_layers?: number[]
  read_load_effective_slots_layers?: number[]
  read_load_max_share_layers?: number[]
  slot_cos_l_last?: number | null
  slot_cos_layers?: number[]
  slot_cos_max_layers?: number[]
  slot_cos_p90_layers?: number[]
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
