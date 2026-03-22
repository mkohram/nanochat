#!/usr/bin/env bash
set -euo pipefail

# Launch the React probe dashboard.
# It serves a live JSON endpoint backed by experiments/out/probe_live.json.

cd "$(dirname "$0")/probe-dashboard"

if [ ! -d node_modules ]; then
  echo "[run_probe_dashboard] installing npm dependencies..."
  npm install
fi

echo "[run_probe_dashboard] dashboard: http://127.0.0.1:4173"
echo "[run_probe_dashboard] watching: ../out/probe_live.json"

exec npm run dev -- --host 127.0.0.1 --port 4173
