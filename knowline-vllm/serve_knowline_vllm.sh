#!/bin/bash
# Serve a KnowLine model behind /v1/systemone on optimized vLLM + vLLM-Ascend 0.30.0 (Ascend 910B4).
#   bash serve_knowline_vllm.sh <model dir> [gpus=0] [port=8080]
# Starts `vllm serve` (port = front-end port + 1000, bound to 127.0.0.1), then knowline_vllm.py: the model dir's own
# knowline_server.py with an optimized vLLM backend (atomic multi-question batching + smart warm skip).
#
# Environment:
#   MEM (0.85)        --gpu-memory-utilization
#   MAXLEN (-1)       --max-model-len; -1 = longest that fits in memory
#   QUANT (auto)      fp8 | bf16; auto defaults to bf16 on Ascend NPU (910B4 does not support native FP8)
#   SEQS (256)        --max-num-seqs
#   IMAGES (8)        images per request (--limit-mm-per-prompt)
#   HOST (127.0.0.1)  front-end bind address
#   VLLM (vllm), PYTHON (python): vLLM CLI and front-end interpreter
#   VLLM_ARGS         extra flags for vllm serve (NOTE: no spaces inside JSON values!)
set -euo pipefail
MODEL=${1:?usage: serve_knowline_vllm.sh <model dir> [gpus] [port]}
GPUS=${2:-0}; PORT=${3:-8080}; VLLM_PORT=$((PORT + 1000)); MEM=${MEM:-0.85}; MAXLEN=${MAXLEN:--1}
QUANT=${QUANT:-auto}; SEQS=${SEQS:-256}; IMAGES=${IMAGES:-8}; HOST=${HOST:-127.0.0.1}; VLLM=${VLLM:-vllm}; PYTHON=${PYTHON:-python}
KIT=$(cd "$(dirname "$0")" && pwd); MODEL=$(cd "$MODEL" && pwd); SERVE="$MODEL/serve_knowline.sh"
[ -f "$SERVE" ] || { echo "no serve_knowline.sh in $MODEL" >&2; exit 1; }

FRONT=$(awk '/knowline_server.py" --model/ {on = 1} on {print; if ($0 !~ /\\$/) exit}' "$SERVE" |
  sed -e 's|^python "$HERE/knowline_server.py"|"$PYTHON" "$KIT/knowline_vllm.py" --server-dir "$HERE"|' \
      -e 's|--backend sglang|--backend vllm|')
case "$FRONT" in *knowline_vllm.py*"--backend vllm"*) ;; *) echo "no knowline_server.py command in $SERVE" >&2; exit 1 ;; esac

if [ "$QUANT" = auto ]; then
  if command -v npu-smi >/dev/null 2>&1 || [ -d /usr/local/Ascend ]; then
    QUANT=bf16
    echo "[knowline-ascend] Ascend NPU detected: defaulting QUANT=bf16 (set QUANT=fp8 to override)" >&2
  elif grep -q -- "--quantization fp8" "$SERVE"; then
    QUANT=fp8
  else
    QUANT=bf16
  fi
fi

ENGINE=(--served-model-name m --host 127.0.0.1 --port "$VLLM_PORT" --tensor-parallel-size "$(awk -F, '{print NF}' <<< "$GPUS")"
  --gpu-memory-utilization "$MEM" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --enable-scale-out --max-logprobs 128
  --generation-config vllm --limit-mm-per-prompt "{\"image\": $IMAGES}")
[ "$QUANT" = fp8 ] && ENGINE+=(--quantization fp8)
grep -q -- "--enable-fp32-lm-head" "$SERVE" && ENGINE+=(--hf-overrides '{"head_dtype": "float32"}')

# HOST-4: Ascend CANN runtime defaults (TASK_QUEUE_ENABLE=2 breaks ACLGraph capture; CPU_AFFINITY_CONF=1 conflicts with cpu_binding)
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-10000}
export HOST_CACHE_CAPACITY=${HOST_CACHE_CAPACITY:-50}
if [ "${TASK_QUEUE_ENABLE}" = "2" ]; then
  echo "[WARN] TASK_QUEUE_ENABLE=2 is incompatible with ACLGraph capture on torch_npu 2.10; recommend TASK_QUEUE_ENABLE=1" >&2
fi
if [ "${CPU_AFFINITY_CONF:-}" = "1" ]; then
  echo "[WARN] CPU_AFFINITY_CONF=1 conflicts with vllm_ascend cpu_binding and regresses latency" >&2
fi

# Full-stack KnowLine optimization defaults (all overridable via env vars)
export VLLM_ASCEND_FP32_HEAD_MM=${VLLM_ASCEND_FP32_HEAD_MM:-1}
export VLLM_ASCEND_GDN_CPP_GLUE=${VLLM_ASCEND_GDN_CPP_GLUE:-1}
export VLLM_ASCEND_GDN_SKIP_UNUSED_FINAL_STATE=${VLLM_ASCEND_GDN_SKIP_UNUSED_FINAL_STATE:-1}
export VLLM_ASCEND_GDN_GATING_IN_GRAPH=${VLLM_ASCEND_GDN_GATING_IN_GRAPH:-1}
export VLLM_ASCEND_SHARE_MROPE_COS_SIN=${VLLM_ASCEND_SHARE_MROPE_COS_SIN:-1}
export KNOWLINE_BATCH_SCORE=${KNOWLINE_BATCH_SCORE:-1}
export KNOWLINE_WARM_MIN_TOKENS=${KNOWLINE_WARM_MIN_TOKENS:-640}
export KNOWLINE_DEFAULT_CG_EXT=${KNOWLINE_DEFAULT_CG_EXT:-1}

# HOST-3: extend Piecewise ACLGraph capture sizes up to 1024 so multi-question atomic batches (e.g. T=384, 644)
# execute inside ACLGraph instead of falling back to Eager.
if [ "${KNOWLINE_DEFAULT_CG_EXT}" != "0" ] && [[ " ${VLLM_ARGS:-} " != *" --compilation-config "* ]]; then
  ENGINE+=(--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128,160,192,200,256,320,384,448,512,576,656,768,896,1024]}')
fi

# shellcheck disable=SC2086  # VLLM_ARGS is a list of flags
CUDA_VISIBLE_DEVICES=$GPUS "$VLLM" serve "$MODEL" "${ENGINE[@]}" ${VLLM_ARGS:-} &
VPID=$!; FPID=
trap 'kill $FPID "$VPID" 2>/dev/null || true' EXIT
trap 'exit 143' TERM INT
until curl -sf -m 3 "http://127.0.0.1:$VLLM_PORT/health" > /dev/null; do
  sleep 5; kill -0 "$VPID" 2>/dev/null || { echo "vLLM exited" >&2; exit 1; }
done
HERE=$MODEL; SGL_PORT=$VLLM_PORT
eval "exec $FRONT" &
FPID=$!
wait "$FPID"
