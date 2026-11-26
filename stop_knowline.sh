#!/usr/bin/env bash
# Cleanly stop KnowLine-4B front-end and vLLM engine processes.
set -uo pipefail

REL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for pf in "$REL_ROOT"/logs/knowline_*.pid; do
  [ -f "$pf" ] || continue
  pid=$(cat "$pf" 2>/dev/null || true)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
  fi
  rm -f "$pf"
done

pkill -f "knowline_vllm.py" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Force-kill only if stubborn EngineCore child lingers
if pgrep -f "knowline_vllm.py|vllm serve" >/dev/null 2>&1; then
  pkill -9 -f "knowline_vllm.py" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 1
fi

echo "KnowLine service stopped. Checking NPU status:"
npu-smi info 2>/dev/null | tail -n 12 || true
