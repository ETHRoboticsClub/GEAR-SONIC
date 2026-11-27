#!/usr/bin/env bash
# Install NVIDIA's Isaac-GR00T at the revision pinned in ISAAC_GR00T_REVISION, for train.py.
#
#   bash ethrc/groot_finetune/setup_groot.sh [directory]     (default: ~/Isaac-GR00T)
#
# Clones https://github.com/NVIDIA/Isaac-GR00T (or reuses the directory if it is already a
# clone), checks out the pinned commit, then runs GR00T's own dGPU install
# (scripts/deployment/dgpu/install_deps.sh: ffmpeg, libaio, the CUDA 12.8 toolkit if
# missing, uv, `uv sync`). Needs an NVIDIA GPU machine with sudo; on a laptop, use the
# cloud launcher (cloud/train.sh) instead. Safe to run again.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
revision="$(tr -d '[:space:]' < "$here/ISAAC_GR00T_REVISION")"
target="${1:-${ISAAC_GR00T_DIR:-$HOME/Isaac-GR00T}}"

if [ -d "$target/.git" ]; then
  git -C "$target" fetch --quiet origin "$revision" || git -C "$target" fetch --quiet origin
else
  git clone --quiet https://github.com/NVIDIA/Isaac-GR00T.git "$target"
fi
if [ -n "$(git -C "$target" status --porcelain --untracked-files=no)" ]; then
  echo "$target has local changes; commit or discard them first" >&2
  exit 1
fi
git -C "$target" -c advice.detachedHead=false checkout --quiet "$revision"
echo "Isaac-GR00T $(git -C "$target" rev-parse --short HEAD) in $target"

# The dGPU wheels are Git LFS files. Without git-lfs the clone holds pointer files in their
# place, and `uv sync` fails reading them ("unable to locate the end of central directory
# record"). uv reads every platform's wheel in the lock, so fetch all of them.
if ! git lfs version > /dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi
git -C "$target" lfs install --local > /dev/null
git -C "$target" lfs pull --include "scripts/deployment/dgpu/wheels/*.whl"

bash "$target/scripts/deployment/dgpu/install_deps.sh"
export PATH="$HOME/.local/bin:$PATH"
(cd "$target" && uv run python -c "import gr00t, torch; print('gr00t ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())")

cat << EOF

Ready. Train with:
  cd $target && uv run python $here/train.py --dataset <namespace/name_gr00t> --model-repo <namespace/model>
EOF
