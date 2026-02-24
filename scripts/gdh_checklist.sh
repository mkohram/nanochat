#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[GDH checklist] Running double-helix oracle tests..."
.venv/bin/python -m pytest -q tests/gdh/test_oracle.py

echo "[GDH checklist] OK"
