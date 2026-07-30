#!/usr/bin/env bash
# Provision the GPU environment inside WSL2.
#
# WHY THIS SCRIPT EXISTS
#
# Windows Smart App Control is enforced on the development machine
# (VerifiedAndReputablePolicyState = 1). PyTorch ships unsigned native libraries,
# so loading it from Windows Python fails with WinError 4551, and the Code
# Integrity event log names torch_cpu.dll as failing the signing requirement --
# which means CPU-only PyTorch is blocked too, not just the CUDA build.
#
# Disabling Smart App Control is not the answer: it is a machine-wide security
# control, and on Windows 11 it cannot be re-enabled without reinstalling the OS.
# WSL2 is the supported route instead. Its Linux userspace is not governed by the
# Windows user-mode code-integrity policy, and the NVIDIA WSL driver exposes the
# GPU through /usr/lib/wsl/lib, so the same hardware is available with no security
# control weakened.
#
# The virtual environment is created on the WSL filesystem rather than under
# /mnt/c. Package installs write tens of thousands of small files, and the 9p
# filesystem that backs /mnt/c makes that roughly an order of magnitude slower.
# Corpus data is still read from /mnt/c, which is fine for a small number of large
# sequential reads.

set -euo pipefail

VENV="${HOME}/ra-gpu"
PROJECT="/mnt/c/Users/sagar/Desktop/retrieval-ablation"

echo "=== environment ==="
python3 --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# uv rather than python3 -m venv. Ubuntu splits ensurepip into a separate
# python3.14-venv package, so the stdlib route needs apt and therefore sudo; uv
# installs to the user's home directory, builds environments without ensurepip,
# and is already the project's package manager on the Windows side.
if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
uv --version

echo "=== creating venv at ${VENV} ==="
# --clear so a partially provisioned environment from an interrupted run is
# replaced rather than reused; a half-installed venv fails later and further away.
uv venv --clear --python 3.14 "${VENV}"
export VIRTUAL_ENV="${VENV}"

# The Linux torch build resolves CUDA runtime libraries as separate nvidia-*
# wheels from pypi.nvidia.com rather than bundling them as the Windows wheel does,
# so this is several gigabytes across a dozen downloads. On a flaky connection at
# least one of them times out even with uv's internal retries, and the failure
# aborts the whole resolution. Retrying at the shell level makes progress each
# attempt because already-downloaded wheels stay in uv's cache.
retry() {
  local attempts=$1
  shift
  local n=1
  until "$@"; do
    if [ "${n}" -ge "${attempts}" ]; then
      echo "FAILED after ${n} attempts: $*" >&2
      return 1
    fi
    n=$((n + 1))
    echo "--- attempt ${n}/${attempts} ---"
    sleep 5
  done
}

echo "=== installing torch (cu126) ==="
retry 6 uv pip install --python "${VENV}/bin/python" torch \
  --index-url https://download.pytorch.org/whl/cu126

echo "=== installing retrieval dependencies ==="
retry 4 uv pip install --python "${VENV}/bin/python" \
  "sentence-transformers>=3.3" "transformers>=4.46"
retry 4 uv pip install --python "${VENV}/bin/python" -e "${PROJECT}"

echo "=== verifying CUDA is actually usable ==="
"${VENV}/bin/python" - <<'PY'
import json
import torch

info = {
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
}
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    info["device"] = props.name
    info["vram_mib"] = props.total_memory // (1024 * 1024)
    info["capability"] = f"{props.major}.{props.minor}"
    # A real allocation and a real matmul: cuda.is_available() can be true while
    # the driver still refuses to run a kernel, and discovering that after an
    # hour of embedding would be expensive.
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    info["matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
print(json.dumps(info, indent=2))
PY

echo "=== done ==="
