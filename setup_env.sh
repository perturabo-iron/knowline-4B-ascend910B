#!/usr/bin/env bash
# One-click bootstrap script for KnowLine-4B optimized vLLM + vLLM-Ascend 0.30.0 on Ascend 910B4.
# Target official base image:
#   https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884?version=9.2.1-910b-ubuntu22.04-py3.12
#
# What this script does automatically:
#   1. Creates an isolated Python 3.12 venv inside ./venv
#   2. Downloads & installs exact pinned versions:
#      torch==2.10.0+cpu, torch-npu==2.10.0.post7, vllm==0.30.0 (empty-device wheel),
#      vllm-ascend==0.30.0 (A2 wheel), triton-ascend==3.6.0.post2,
#      flash-linear-attention-npu-a2==26.9.1, arctic-inference==0.1.1
#   3. Automatically fixes upstream wheel metadata conflicts (vllm-ascend fastapi pin &
#      triton-ascend numpy/triton conflict) without touching any binary code
#   4. Applies full optimization patches (patches/vllm_0.30.0_full.patch & vllm_ascend_0.30.0_full.patch)
#   5. Compiles the GDN Core C++ glue extension (_gdn_core_C.so) and runs import self-check
#
# Usage:
#   bash setup_env.sh              # Full install + patch + compile
#   bash setup_env.sh --rebuild    # Remove existing ./venv and rebuild from scratch
#   bash setup_env.sh --patch-only # Reuse existing ./venv, only (re-)apply patches & build _gdn_core_C.so
#
# Optional env vars:
#   PYTHON_BIN=/usr/local/python3.12.16/bin/python3
#   PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
#   PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
#   ASCEND_PYPI_BASE=https://mirrors.huaweicloud.com/ascend/repos/pypi
#   KEEP_WHEELS=1   (keep temporary .build_wheels/ cache after install)
set -euo pipefail

REL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="$REL_ROOT/venv"
BUILD_DIR="$REL_ROOT/.build_wheels"

PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}
PYTORCH_CPU_INDEX=${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}
ASCEND_PYPI_BASE=${ASCEND_PYPI_BASE:-https://mirrors.huaweicloud.com/ascend/repos/pypi}

MODE="full"
for arg in "$@"; do
  case "$arg" in
    --rebuild)    MODE="rebuild" ;;
    --patch-only) MODE="patch-only" ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# 1. Pre-flight toolchain & CANN checks
for cmd in g++ patch curl tar; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: required command '$cmd' not found in PATH" >&2; exit 1; }
done

if [ ! -f /usr/local/Ascend/cann-9.2.1/set_env.sh ] && [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  echo "ERROR: Ascend CANN not found under /usr/local/Ascend/. Please run inside the AscendHub 9.2.1-910b-ubuntu22.04-py3.12 image." >&2
  exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  echo "[WARN] Detected architecture '$ARCH' (expected 'aarch64' / Kunpeng-920 on Ascend 910B4)." >&2
fi

pick_python312() {
  if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
    echo "$PYTHON_BIN"; return 0
  fi
  for cand in /usr/local/python3.12.16/bin/python3 python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      local ver
      ver=$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
      if [ "$ver" = "3.12" ]; then
        command -v "$cand"; return 0
      fi
    fi
  done
  return 1
}

if [ "$MODE" = "rebuild" ] && [ -d "$VENV_DIR" ]; then
  echo "=== Removing existing venv: $VENV_DIR ==="
  rm -rf "$VENV_DIR"
fi

if [ "$MODE" != "patch-only" ]; then
  PY312=$(pick_python312) || { echo "ERROR: Python 3.12 not found. Set PYTHON_BIN=/path/to/python3.12" >&2; exit 1; }
  echo "=== [1/6] Creating Python 3.12 venv using $PY312 ==="
  if [ ! -d "$VENV_DIR" ]; then
    "$PY312" -m venv "$VENV_DIR"
  fi

  VENV_PY="$VENV_DIR/bin/python"
  VENV_PIP="$VENV_DIR/bin/pip"
  mkdir -p "$BUILD_DIR"/{sdist,wheels,repacked}

  echo "=== [2/6] Installing build toolchain & torch==2.10.0+cpu ==="
  env -u PYTHONPATH "$VENV_PIP" install --upgrade \
    -i "$PIP_INDEX_URL" \
    "pip==25.0.1" "wheel==0.48.0" "uv==0.13.9" \
    "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" \
    "packaging>=24.2" "cmake>=3.26.1" "ninja" "jinja2" "nanobind==2.9.2" "grpcio-tools"

  TORCH_WHL_NAME="torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl"
  if [ ! -f "$BUILD_DIR/wheels/$TORCH_WHL_NAME" ]; then
    env -u PYTHONPATH "$VENV_PIP" download --no-deps \
      -i "$PYTORCH_CPU_INDEX" \
      -d "$BUILD_DIR/wheels" "torch==2.10.0+cpu"
  fi
  TORCH_WHL=$(ls "$BUILD_DIR/wheels"/torch-2.10.0*.whl | head -n 1)
  env -u PYTHONPATH "$VENV_PIP" install --no-deps "$TORCH_WHL"

  echo "=== [3/6] Downloading & building vllm==0.30.0 (empty device) and arctic-inference==0.1.1 ==="
  if ! ls "$BUILD_DIR/wheels"/vllm-0.30.0-*.whl >/dev/null 2>&1; then
    env -u PYTHONPATH "$VENV_PIP" download --no-deps --no-binary vllm \
      -i "$PIP_INDEX_URL" -d "$BUILD_DIR/sdist" "vllm==0.30.0"
    rm -rf "$BUILD_DIR/sdist/vllm-0.30.0"
    tar -xzf "$BUILD_DIR/sdist/vllm-0.30.0.tar.gz" -C "$BUILD_DIR/sdist"
    VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0 \
      env -u PYTHONPATH "$VENV_PY" -m pip wheel --no-deps --no-build-isolation \
      -w "$BUILD_DIR/wheels" "$BUILD_DIR/sdist/vllm-0.30.0"
  fi

  if ! ls "$BUILD_DIR/wheels"/arctic_inference-0.1.1-*.whl >/dev/null 2>&1; then
    env -u PYTHONPATH "$VENV_PIP" download --no-deps --no-binary arctic-inference \
      -i "$PIP_INDEX_URL" -d "$BUILD_DIR/sdist" "arctic-inference==0.1.1"
    rm -rf "$BUILD_DIR/sdist/arctic_inference-0.1.1"
    tar -xzf "$BUILD_DIR/sdist/arctic_inference-0.1.1.tar.gz" -C "$BUILD_DIR/sdist"
    env -u PYTHONPATH "$VENV_PY" -m pip wheel --no-deps --no-build-isolation \
      -w "$BUILD_DIR/wheels" "$BUILD_DIR/sdist/arctic_inference-0.1.1"
  fi

  echo "=== [4/6] Downloading Ascend wheels & fixing upstream metadata conflicts ==="
  # 4a. vllm-ascend==0.30.0 (cp312 aarch64)
  if ! ls "$BUILD_DIR/wheels"/vllm_ascend-0.30.0-*.whl >/dev/null 2>&1; then
    env -u PYTHONPATH "$VENV_PIP" download --no-deps --only-binary=:all: \
      -i "$PIP_INDEX_URL" -d "$BUILD_DIR/wheels" "vllm-ascend==0.30.0"
  fi

  # 4b. triton-ascend==3.6.0.post2 & flash-linear-attention-npu-a2==26.9.1 from Huawei Ascend PyPI
  TRITON_WHL_NAME="triton_ascend-3.6.0.post2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
  FLA_WHL_NAME="flash_linear_attention_npu_a2-26.9.1-py3-none-manylinux_2_34_aarch64.whl"
  if [ ! -f "$BUILD_DIR/wheels/$TRITON_WHL_NAME" ]; then
    curl -fL --retry 3 -o "$BUILD_DIR/wheels/$TRITON_WHL_NAME" \
      "${ASCEND_PYPI_BASE}/triton-ascend/${TRITON_WHL_NAME}"
  fi
  if [ ! -f "$BUILD_DIR/wheels/$FLA_WHL_NAME" ]; then
    curl -fL --retry 3 -o "$BUILD_DIR/wheels/$FLA_WHL_NAME" \
      "${ASCEND_PYPI_BASE}/flash-linear-attention-npu-a2/${FLA_WHL_NAME}"
  fi

  # 4c. Repack vllm-ascend (fastapi range) & triton-ascend (numpy>=2,<3 + remove conflicting triton==3.6.0)
  env -u PYTHONPATH "$VENV_PY" - "$BUILD_DIR" <<'PYEOF'
import glob, os, shutil, subprocess, sys

build_dir = sys.argv[1]
repacked = os.path.join(build_dir, "repacked")
tmp_unpack = os.path.join(build_dir, "unpack")
shutil.rmtree(tmp_unpack, ignore_errors=True)
os.makedirs(tmp_unpack, exist_ok=True)
os.makedirs(repacked, exist_ok=True)

def repack_wheel(whl_path, transform_fn):
    out_whl = os.path.join(repacked, os.path.basename(whl_path))
    if os.path.isfile(out_whl):
        return out_whl
    work = os.path.join(tmp_unpack, os.path.basename(whl_path) + ".d")
    shutil.rmtree(work, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "wheel", "unpack", "-d", work, whl_path], check=True)
    roots = [os.path.join(work, d) for d in os.listdir(work) if os.path.isdir(os.path.join(work, d))]
    assert len(roots) == 1, f"Unexpected wheel root dirs: {roots}"
    root = roots[0]
    metas = glob.glob(os.path.join(root, "*.dist-info", "METADATA"))
    assert len(metas) == 1, f"METADATA not found in {root}"
    with open(metas[0], "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = transform_fn(lines)
    with open(metas[0], "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    subprocess.run([sys.executable, "-m", "wheel", "pack", "-d", repacked, root], check=True)
    shutil.rmtree(work, ignore_errors=True)
    return out_whl

ascend_whl = sorted(glob.glob(os.path.join(build_dir, "wheels", "vllm_ascend-0.30.0-*.whl")))[0]
triton_whl = sorted(glob.glob(os.path.join(build_dir, "wheels", "triton_ascend-3.6.0.post2-*.whl")))[0]

def fix_ascend(lines):
    return [
        "Requires-Dist: fastapi>=0.133.0,<0.137.0\n" if l.strip() == "Requires-Dist: fastapi<0.124.0" else l
        for l in lines
    ]

def fix_triton(lines):
    out = []
    for l in lines:
        s = l.strip()
        if s == "Requires-Dist: numpy==1.26.4":
            out.append("Requires-Dist: numpy>=2,<3\n")
        elif s == "Requires-Dist: triton==3.6.0":
            continue  # Standalone triton overwrites triton-ascend's triton module
        else:
            out.append(l)
    return out

print("Repacked:", repack_wheel(ascend_whl, fix_ascend))
print("Repacked:", repack_wheel(triton_whl, fix_triton))
shutil.rmtree(tmp_unpack, ignore_errors=True)
PYEOF

  echo "=== [5/6] Resolving & installing full package set with uv ==="
  VLLM_WHL=$(ls "$BUILD_DIR/wheels"/vllm-0.30.0-*.whl | head -n 1)
  ARCTIC_WHL=$(ls "$BUILD_DIR/wheels"/arctic_inference-0.1.1-*.whl | head -n 1)
  ASCEND_WHL=$(ls "$BUILD_DIR/repacked"/vllm_ascend-0.30.0-*.whl | head -n 1)
  TRITON_WHL=$(ls "$BUILD_DIR/repacked"/triton_ascend-3.6.0.post2-*.whl | head -n 1)
  FLA_WHL="$BUILD_DIR/wheels/$FLA_WHL_NAME"

  env -u PYTHONPATH "$VENV_DIR/bin/uv" pip install \
    --python "$VENV_PY" \
    --index-url "$PIP_INDEX_URL" \
    --extra-index-url "$PYTORCH_CPU_INDEX" \
    --index-strategy unsafe-best-match \
    -c "$REL_ROOT/constraints.txt" \
    "$TORCH_WHL" \
    "$VLLM_WHL" \
    "$ASCEND_WHL" \
    "$TRITON_WHL" \
    "$FLA_WHL" \
    "$ARCTIC_WHL" \
    "torch-npu==2.10.0.post7" \
    "transformers==5.14.1" \
    "requests"

  # Ensure standalone triton is never shadowing triton_ascend
  if env -u PYTHONPATH "$VENV_PIP" show triton >/dev/null 2>&1; then
    env -u PYTHONPATH "$VENV_PIP" uninstall -y triton
    env -u PYTHONPATH "$VENV_PIP" install --no-deps --force-reinstall "$TRITON_WHL"
  fi

  env -u PYTHONPATH "$VENV_PY" -m pip check

  if [ "${KEEP_WHEELS:-0}" != "1" ]; then
    rm -rf "$BUILD_DIR"
  fi
fi

echo "=== [6/6] Applying full optimization patches & compiling _gdn_core_C.so ==="
source "$REL_ROOT/activate.sh"
SITE_PKG=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

# Apply patches idempotently (skip if already applied)
if patch -p1 --batch -R --dry-run -d "$SITE_PKG" -i "$REL_ROOT/patches/vllm_0.30.0_full.patch" >/dev/null 2>&1; then
  echo "vllm_0.30.0_full.patch already applied, skipping."
else
  patch -p1 --batch -d "$SITE_PKG" -i "$REL_ROOT/patches/vllm_0.30.0_full.patch"
fi

if patch -p1 --batch -R --dry-run -d "$SITE_PKG" -i "$REL_ROOT/patches/vllm_ascend_0.30.0_full.patch" >/dev/null 2>&1; then
  echo "vllm_ascend_0.30.0_full.patch already applied, skipping."
else
  patch -p1 --batch -d "$SITE_PKG" -i "$REL_ROOT/patches/vllm_ascend_0.30.0_full.patch"
fi

bash "$REL_ROOT/csrc/build_gdn_core_glue.sh"

echo "=== Verifying complete environment imports ==="
python3 - <<'PYEOF'
import torch, torch_npu, triton, fla_npu, vllm, vllm_ascend
from vllm_ascend import _gdn_core_C
print("SETUP_ALL_OK")
print("  torch       :", torch.__version__)
print("  torch_npu   :", torch_npu.__version__)
print("  vllm        :", vllm.__version__, vllm.__file__)
print("  vllm_ascend :", vllm_ascend.__file__)
print("  _gdn_core_C :", _gdn_core_C.__file__, [a for a in dir(_gdn_core_C) if not a.startswith("_")])
PYEOF

echo ""
echo "Setup finished! Next step:"
echo "  BG=1 bash $REL_ROOT/start_knowline.sh /path/to/KnowLine-4B-Gen5.5"
