#!/bin/bash
# Install video_to_trajectory into the active Python 3.10 environment.
#
#   python3.10 -m venv venv && source venv/bin/activate && ./install.sh
#
# Python 3.10 is required: DPVO's CUDA extensions are built against torch 2.3.1,
# which has no wheels for newer Pythons.
#
# Set CUDA_TAG for your driver's CUDA version (cu118, cu121, cu124, ...):
#   CUDA_TAG=cu124 ./install.sh
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu121}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPVO="$ROOT/third_party/dpvo"
EIGEN_VERSION="3.4.0"

echo "==> Python $(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'), CUDA tag $CUDA_TAG"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    sys.exit(f"Python 3.10 is required (DPVO pins torch 2.3.1); this is "
             f"{sys.version.split()[0]}")
PY

# 1 - torch first: everything else builds against it.
echo "==> installing torch"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    torch==2.3.1 torchvision

# 2 - torch-scatter compiles against the torch just installed, so it must not
#     be built in an isolated environment that would fetch a different one.
echo "==> installing torch-scatter"
pip install --no-build-isolation torch-scatter

# 3 - everything else.
echo "==> installing requirements"
pip install -r "$ROOT/requirements.txt"

# 4 - Eigen, a header-only build dependency of DPVO's bundle-adjustment kernels.
if [ ! -d "$DPVO/thirdparty/eigen-$EIGEN_VERSION" ]; then
    echo "==> downloading Eigen $EIGEN_VERSION"
    mkdir -p "$DPVO/thirdparty"
    tmp="$(mktemp -d)"
    wget -q --show-progress -O "$tmp/eigen.zip" \
        "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.zip"
    unzip -q "$tmp/eigen.zip" -d "$DPVO/thirdparty"
    rm -rf "$tmp"
else
    echo "==> Eigen already present"
fi

# 5 - If nvcc and your system compiler disagree with the one torch was built
#     with, pin them here. gcc-12 is the usual answer on Ubuntu 24.04.
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12

# 6 - Build the patched DPVO (three CUDA extensions; this takes a few minutes).
if python -c "import dpvo" 2>/dev/null; then
    echo "==> DPVO already installed"
else
    echo "==> building DPVO (this takes a few minutes)"
    pip install --no-build-isolation "$DPVO"
fi

# 7 - DPVO's pretrained weights.
if [ ! -f "$DPVO/dpvo.pth" ]; then
    echo "==> downloading DPVO weights"
    tmp="$(mktemp -d)"
    wget -q --show-progress -O "$tmp/models.zip" \
        "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
    unzip -q -j "$tmp/models.zip" "dpvo.pth" -d "$DPVO"
    rm -rf "$tmp"
else
    echo "==> DPVO weights already present"
fi

echo
echo "==> done. Check it with:"
echo "    python -c 'import dpvo, torch; print(torch.cuda.is_available())'"
echo "    python video_to_trajectory.py --help"
echo
echo "Metric3D's weights download themselves on first use."
echo "For DepthPro instead, see requirements-depthpro.txt."
