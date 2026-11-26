# Source this script from Bash to activate the KnowLine Ascend 910B4 vLLM environment.
# Usage: source /path/to/knowline-ascend-release/activate.sh
_knowline_rel_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

_knowline_was_nounset=0
case $- in
  *u*) _knowline_was_nounset=1; set +u ;;
esac

source "$_knowline_rel_root/venv/bin/activate"
unset PYTHONPATH
if [ -f /usr/local/Ascend/cann-9.2.1/set_env.sh ]; then
  source /usr/local/Ascend/cann-9.2.1/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export VLLM_PLUGINS=${VLLM_PLUGINS-ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model}
export PYTHONNOUSERSITE=1

# Verified Ascend CANN runtime defaults (do NOT set TASK_QUEUE_ENABLE=2 or CPU_AFFINITY_CONF=1)
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-10000}
export HOST_CACHE_CAPACITY=${HOST_CACHE_CAPACITY:-50}

if [ "$_knowline_was_nounset" -eq 1 ]; then
  set -u
fi
unset _knowline_rel_root _knowline_was_nounset
