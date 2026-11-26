#!/usr/bin/env bash
# Apply all full release patches onto a clean vllm 0.30.0 + vllm-ascend 0.30.0 site-packages directory
# and compile the GDN Core C++ glue extension (_gdn_core_C.so).
# Usage: bash apply_release_patches.sh <target_site_packages_dir>
set -euo pipefail

TARGET_SITE=${1:?usage: apply_release_patches.sh <target_site_packages_dir>}
PATCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REL_ROOT=$(cd -- "$PATCH_DIR/.." && pwd)

[ -d "$TARGET_SITE/vllm" ] || { echo "ERROR: $TARGET_SITE/vllm not found" >&2; exit 1; }
[ -d "$TARGET_SITE/vllm_ascend" ] || { echo "ERROR: $TARGET_SITE/vllm_ascend not found" >&2; exit 1; }

echo "=== Dry-run checking vllm_0.30.0_full.patch ==="
patch -p1 --batch --dry-run -d "$TARGET_SITE" -i "$PATCH_DIR/vllm_0.30.0_full.patch"
echo "=== Dry-run checking vllm_ascend_0.30.0_full.patch ==="
patch -p1 --batch --dry-run -d "$TARGET_SITE" -i "$PATCH_DIR/vllm_ascend_0.30.0_full.patch"

echo "=== Applying patches ==="
patch -p1 --batch -d "$TARGET_SITE" -i "$PATCH_DIR/vllm_0.30.0_full.patch"
patch -p1 --batch -d "$TARGET_SITE" -i "$PATCH_DIR/vllm_ascend_0.30.0_full.patch"

echo "=== Compiling _gdn_core_C.so into $TARGET_SITE ==="
TARGET_SITE_PKG="$TARGET_SITE" bash "$REL_ROOT/csrc/build_gdn_core_glue.sh"
echo "=== All release patches applied and C++ glue built successfully ==="
