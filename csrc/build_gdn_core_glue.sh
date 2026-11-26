#!/usr/bin/env bash
set -eo pipefail

CSRC_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REL_ROOT=$(cd -- "$CSRC_DIR/.." && pwd)

if [ -n "${TARGET_SITE_PKG:-}" ]; then
  SITE_PKG="$TARGET_SITE_PKG"
else
  source "$REL_ROOT/activate.sh"
  SITE_PKG=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
fi

SRC_FILE="$CSRC_DIR/gdn_core_glue.cpp"
EXT_SUFFIX=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
OUT_SO="$SITE_PKG/vllm_ascend/_gdn_core_C${EXT_SUFFIX}"

PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
TORCH_INC="$SITE_PKG/torch/include"
TORCH_API_INC="$SITE_PKG/torch/include/torch/csrc/api/include"
TORCH_LIB="$SITE_PKG/torch/lib"
TORCH_NPU_INC="$SITE_PKG/torch_npu/include"
TORCH_NPU_LIB="$SITE_PKG/torch_npu/lib"
ASCEND_INC="${ASCEND_HOME_PATH}/include"
ASCEND_LIB="${ASCEND_HOME_PATH}/lib64"
CXX11_ABI=$(python3 -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')

echo "Compiling $SRC_FILE -> $OUT_SO (CXX11_ABI=$CXX11_ABI)"
g++ -O3 -std=c++17 -shared -fPIC \
    -DTORCH_EXTENSION_NAME=_gdn_core_C \
    -D_GLIBCXX_USE_CXX11_ABI="${CXX11_ABI}" \
    -I"${PY_INC}" \
    -I"${TORCH_INC}" \
    -I"${TORCH_API_INC}" \
    -I"${TORCH_NPU_INC}" \
    -I"${ASCEND_INC}" \
    "${SRC_FILE}" \
    -L"${TORCH_LIB}" \
    -L"${TORCH_NPU_LIB}" \
    -L"${ASCEND_LIB}" \
    -Wl,-rpath,'$ORIGIN/../torch/lib' \
    -Wl,-rpath,'$ORIGIN/../torch_npu/lib' \
    -Wl,-rpath,"${ASCEND_LIB}" \
    -ltorch -ltorch_cpu -ltorch_python -lc10 -ltorch_npu -lascendcl \
    -o "${OUT_SO}"

echo "Build succeeded: $(ls -lh "${OUT_SO}")"
python3 -c 'import torch, torch_npu, vllm_ascend._gdn_core_C as m; print("Loaded _gdn_core_C:", m.__file__); print("Exports:", [a for a in dir(m) if not a.startswith("_")]); print("Torch op:", torch.ops._vllm_ascend_gdn.gdn_prefill_core)'
