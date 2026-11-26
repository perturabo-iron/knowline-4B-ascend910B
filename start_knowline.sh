#!/usr/bin/env bash
# One-click launcher for optimized KnowLine-4B on Ascend 910B4.
# Usage:
#   bash start_knowline.sh [model_dir=$HOME/work/knowline] [npu_id=0] [port=8080]
# Options via env vars:
#   BG=1         Start in background (nohup) and wait until /health is ready
#   STRICT_B1=1  Strict B=1 audit mode (disables batching & warm skip, max diff < 1e-6)
#   QUANT=bf16   Precision override (default: bf16 on Ascend)
#   HOST=0.0.0.0 Front-end bind address (default: 127.0.0.1)
set -euo pipefail

REL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${1:-${KNOWLINE_MODEL_DIR:-$HOME/work/knowline}}
GPUS=${2:-0}
PORT=${3:-8080}

if [ ! -d "$REL_ROOT/venv" ]; then
  echo "ERROR: $REL_ROOT/venv not found!" >&2
  echo "Please initialize the optimized environment first:" >&2
  echo "  bash $REL_ROOT/setup_env.sh" >&2
  exit 1
fi

if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/serve_knowline.sh" ]; then
  echo "ERROR: Invalid KnowLine model directory: $MODEL_DIR (missing serve_knowline.sh)" >&2
  echo "Usage: bash $0 <model_dir> [npu_id=0] [port=8080]" >&2
  echo "Tip: download model via:" >&2
  echo "  git clone https://gitcode.com/hf_mirrors/PelaAI/KnowLine-4B-Gen5.5.git \$HOME/work/knowline" >&2
  exit 1
fi

# Activate self-contained vLLM + vllm-ascend + CANN + ATB environment
source "$REL_ROOT/activate.sh"

export QUANT=${QUANT:-bf16}
if [ "${STRICT_B1:-0}" = "1" ]; then
  export KNOWLINE_BATCH_SCORE=0
  export KNOWLINE_WARM_MIN_TOKENS=0
fi

# Verify C++ glue exists before launch
SO_PATH=$(ls "$REL_ROOT"/venv/lib/python*/site-packages/vllm_ascend/_gdn_core_C*.so 2>/dev/null | head -n 1 || true)
if [ -z "$SO_PATH" ]; then
  echo "[WARN] _gdn_core_C.so not found, rebuilding automatically..." >&2
  bash "$REL_ROOT/csrc/build_gdn_core_glue.sh"
  SO_PATH=$(ls "$REL_ROOT"/venv/lib/python*/site-packages/vllm_ascend/_gdn_core_C*.so 2>/dev/null | head -n 1 || true)
fi

echo "============================================================"
echo " KnowLine-4B Optimized Ascend 910B4 Service"
echo "------------------------------------------------------------"
echo " Release Root : $REL_ROOT"
echo " Python / vLLM: $(which python3) | $(which vllm)"
echo " GDN C++ Glue : ${SO_PATH:-MISSING}"
echo " Model Dir    : $MODEL_DIR"
echo " Device / Port: NPU=$GPUS | Front=http://${HOST:-127.0.0.1}:${PORT} | Engine=127.0.0.1:$((PORT + 1000))"
echo " Mode         : QUANT=$QUANT | STRICT_B1=${STRICT_B1:-0} | BATCH=${KNOWLINE_BATCH_SCORE:-1} | WARM_MIN=${KNOWLINE_WARM_MIN_TOKENS:-640} | CG_EXT=${KNOWLINE_DEFAULT_CG_EXT:-1}"
echo " Opt Switches : FP32_HEAD_MM=${VLLM_ASCEND_FP32_HEAD_MM:-1} | CPP_GLUE=${VLLM_ASCEND_GDN_CPP_GLUE:-1} | SKIP_FS=${VLLM_ASCEND_GDN_SKIP_UNUSED_FINAL_STATE:-1} | GATING_CG=${VLLM_ASCEND_GDN_GATING_IN_GRAPH:-1} | SHARE_MROPE=${VLLM_ASCEND_SHARE_MROPE_COS_SIN:-1}"
echo "============================================================"

if [ "${BG:-0}" = "1" ]; then
  mkdir -p "$REL_ROOT/logs"
  LOG_FILE="$REL_ROOT/logs/knowline_${PORT}.log"
  PID_FILE="$REL_ROOT/logs/knowline_${PORT}.pid"
  nohup bash "$REL_ROOT/knowline-vllm/serve_knowline_vllm.sh" "$MODEL_DIR" "$GPUS" "$PORT" \
    > "$LOG_FILE" 2>&1 </dev/null &
  SRV_PID=$!
  echo "$SRV_PID" > "$PID_FILE"
  echo "Started in background (PID=$SRV_PID), log: $LOG_FILE"
  echo "Waiting for http://127.0.0.1:${PORT}/health ..."
  for i in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "READY! Service is live on http://${HOST:-127.0.0.1}:${PORT}/v1/systemone (took ~$((i * 3))s)"
      echo "Run verification: $REL_ROOT/venv/bin/python $REL_ROOT/test_knowline.py --url http://127.0.0.1:${PORT}"
      exit 0
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "ERROR: Service exited unexpectedly during startup! Last 40 lines of $LOG_FILE:" >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      exit 1
    fi
    if [ $((i % 10)) -eq 0 ]; then
      echo "  still initializing ($((i * 3))s)..."
    fi
    sleep 3
  done
  echo "ERROR: Timed out waiting for /health after 360s. Check $LOG_FILE" >&2
  exit 1
else
  exec bash "$REL_ROOT/knowline-vllm/serve_knowline_vllm.sh" "$MODEL_DIR" "$GPUS" "$PORT"
fi
