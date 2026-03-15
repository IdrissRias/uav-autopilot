#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/src/uav/config/dev_fast.yaml"

# Run N short flights back-to-back.
N=${1:-10}
DUR=${2:-60}

for i in $(seq 1 "$N"); do
  echo "\n=== RUN $i/$N (duration ${DUR}s) ==="
  "$PY" -u -m uav.main --config "$CFG" --duration-s "$DUR" || true
  sleep 2
done
