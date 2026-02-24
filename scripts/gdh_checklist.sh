#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[GDH checklist] Running GDH oracle test suite..."
.venv/bin/python -m pytest -q tests/gdh

echo "[GDH checklist] OK"
