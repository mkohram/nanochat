#!/usr/bin/env bash
set -euo pipefail

# Canonical HARD blindfold probe config for Apple Silicon / MPS.
# Uses the MPS-tuned lab harness, defaults MPS autocast to bfloat16,
# enables torch.compile, and logs every 50 steps by default.
# Probe defaults now use the simplified GDH path:
# - read-mute gate off
# - write brain off
# - single write head
# Pass flags through "$@" to re-enable ablations.

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
OUT_DIR="experiments/out"
mkdir -p "$OUT_DIR"

STEPS=5000
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  case "$arg" in
    --steps)
      j=$((i + 1))
      if [ $j -le $# ]; then
        STEPS="${!j}"
      fi
      ;;
    --steps=*)
      STEPS="${arg#--steps=}"
      ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/probe_hard_mps_${STEPS}_${TS}.log"
LIVE_JSON="${OUT_DIR}/probe_live.json"

echo "$LOG" > "${OUT_DIR}/probe_current_log_path.txt"

echo "[run_probe_hard_mps] log: $LOG"
echo "[run_probe_hard_mps] live_json: $LIVE_JSON"

exec "$PY" experiments/mqar_gdh_mps_lab.py \
  --arch gdh \
  --betas 1.0 \
  --steps 5000 \
  --log-every 50 \
  --seed 123 \
  --sequence-len 256 \
  --vocab-size 128 \
  --n-layer 4 \
  --n-head 8 \
  --n-embd 128 \
  --gdh-slots 8 \
  --gdh-write-heads 1 \
  --route-topk 4 \
  --write-routing static \
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
  --device mps \
  --amp-dtype auto \
  --compile \
  --enforce-capacity-stress \
  --live-json "$LIVE_JSON" \
  "$@" | tee "$LOG"
