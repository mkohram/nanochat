#!/usr/bin/env bash
set -euo pipefail

# Canonical HARD blindfold probe config.
# Harder MQAR: longer sequence + larger gap + n_pairs > slots (capacity stress).

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
OUT_DIR="experiments/out"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/probe_hard_5k_${TS}.log"
LIVE_JSON="${OUT_DIR}/probe_live.json"

echo "$LOG" > "${OUT_DIR}/probe_current_log_path.txt"

echo "[run_probe_hard] log: $LOG"
echo "[run_probe_hard] live_json: $LIVE_JSON"

exec "$PY" experiments/mqar_gdh_lab.py \
  --arch gdh \
  --betas 1.0 \
  --steps 5000 \
  --log-every 10 \
  --seed 123 \
  --sequence-len 256 \
  --vocab-size 128 \
  --n-layer 4 \
  --n-head 8 \
  --n-embd 128 \
  --gdh-slots 8 \
  --gdh-write-heads 1 \
  --gdh-no-write-brain \
  --route-topk 4 \
  --usage-balance-lambda 0.01 \
  --swa-window 8 \
  --n-pairs 16 \
  --n-queries 8 \
  --gap-min 64 \
  --gap-max 192 \
  --key-vocab 32 \
  --value-vocab 32 \
  --query-offset 64 \
  --filler-offset 96 \
  --filler-vocab 32 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --eval-topk 5 \
  --lr 3e-4 \
  --enforce-capacity-stress \
  --live-json "$LIVE_JSON" \
  "$@" | tee "$LOG"
