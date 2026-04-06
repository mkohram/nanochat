#!/usr/bin/env bash
set -euo pipefail

# Canonical HARD ClimbMix probe config for Apple Silicon / MPS.
# Wraps the active hard MPS launcher and applies the real-text overrides used
# for current ClimbMix stability checks.
#
# Overrides:
# - data source: ClimbMix
# - unblinded attention: swa_window=0
# - larger GDH budget: 32 slots, 16 write heads
# - future-summary auxiliary loss off
# - smaller eval batch and denser logging for quicker stability reads
#
# Pass flags through "$@" to override any of the settings below.

cd "$(dirname "$0")/.."

exec bash experiments/run_probe_hard_mps.sh \
  --data-source climbmix \
  --gdh-slots 32 \
  --gdh-write-heads 16 \
  --swa-window 0 \
  --future-summary-horizon 0 \
  --future-summary-lambda 0.0 \
  --eval-batch-size 2 \
  --log-every 5 \
  "$@"
