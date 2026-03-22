#!/usr/bin/env bash
set -euo pipefail

# Canonical EASY blindfold probe config for Apple Silicon / MPS.
# Uses the MPS-tuned lab harness, defaults MPS autocast to bfloat16,
# enables torch.compile, and logs every 50 steps by default.

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
OUT_DIR="experiments/out"
mkdir -p "$OUT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/probe_easy_mps_5k_${TS}.log"
LIVE_JSON="${OUT_DIR}/probe_live.json"

echo "$LOG" > "${OUT_DIR}/probe_current_log_path.txt"

echo "[run_probe_easy_mps] log: $LOG"
echo "[run_probe_easy_mps] live_json: $LIVE_JSON"

exec "$PY" experiments/mqar_gdh_mps_lab.py \
  --arch gdh \
  --betas 1.0 \
  --steps 5000 \
  --log-every 50 \
  --seed 123 \
  --sequence-len 64 \
  --vocab-size 128 \
  --n-layer 3 \
  --n-head 4 \
  --n-embd 64 \
  --gdh-slots 8 \
  --gdh-write-heads 8 \
  --route-topk 4 \
  --usage-balance-lambda 0.01 \
  --swa-window 8 \
  --n-pairs 2 \
  --n-queries 2 \
  --gap-min 16 \
  --gap-max 32 \
  --key-vocab 32 \
  --value-vocab 32 \
  --query-offset 64 \
  --filler-offset 96 \
  --filler-vocab 32 \
  --batch-size 32 \
  --eval-batch-size 8 \
  --eval-topk 5 \
  --lr 3e-4 \
  --device mps \
  --amp-dtype auto \
  --compile \
  --live-json "$LIVE_JSON" \
  "$@" | tee "$LOG"
