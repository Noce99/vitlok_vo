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

# 1 - pip first. The pip that Python 3.10's venv bundles (23.0.1) rejects wheels
#     whose metadata name uses an underscore while the requirement uses a hyphen
#     ("typing_extensions" vs "typing-extensions"), falls back to the sdist, and
#     then cannot reach PyPI for its build dependencies because step 2 restricts
#     the index to PyTorch's. pip 23.3+ normalises the two names and is fine.
echo "==> upgrading pip"
pip install --quiet --upgrade "pip>=23.3" setuptools wheel

# 2 - torch next: everything else builds against it. torchvision is pinned to the
#     release that pairs with torch 2.3.1; leaving it loose makes pip download a
#     newer torchvision, discover it wants a newer torch, and backtrack.
echo "==> installing torch"
pip install --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    torch==2.3.1 torchvision==0.18.1

# 3 - torch-scatter compiles against the torch just installed, so it must not
#     be built in an isolated environment that would fetch a different one.
echo "==> installing torch-scatter"
pip install --no-build-isolation torch-scatter

# 4 - everything else.
echo "==> installing requirements"
pip install -r "$ROOT/requirements.txt"

# 5 - Eigen, a header-only build dependency of DPVO's bundle-adjustment kernels.
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

# 6 - If nvcc and your system compiler disagree with the one torch was built
#     with, pin them here. gcc-12 is the usual answer on Ubuntu 24.04.
# export CC=/usr/bin/gcc-12
# export CXX=/usr/bin/g++-12

# 7 - Build the patched DPVO (three CUDA extensions; this takes a few minutes).
if python -c "import dpvo" 2>/dev/null; then
    echo "==> DPVO already installed"
else
    echo "==> building DPVO (this takes a few minutes)"
    pip install --no-build-isolation "$DPVO"
fi

# 8 - DPVO's pretrained weights.
#
#     Upstream's download_models_and_data.sh points at a Dropbox link that is now
#     dead — it serves an HTML error page, which unzip then rejects with
#     "End-of-central-directory signature not found". We try a Hugging Face
#     mirror first and keep Dropbox as a fallback in case it comes back. Every
#     candidate is checksummed against the known-good file before being installed,
#     so an error page can never be mistaken for weights again.
#
#     Already have the file? Point DPVO_WEIGHTS at it to skip the download:
#       DPVO_WEIGHTS=/path/to/dpvo.pth ./install.sh
DPVO_PTH_SHA256="30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"

check_sha256() {  # check_sha256 FILE -> 0 if it matches the expected digest
    [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$DPVO_PTH_SHA256" ]
}

if [ -f "$DPVO/dpvo.pth" ]; then
    echo "==> DPVO weights already present"
elif [ -n "${DPVO_WEIGHTS:-}" ]; then
    echo "==> using DPVO weights from \$DPVO_WEIGHTS"
    check_sha256 "$DPVO_WEIGHTS" \
        || echo "    warning: $DPVO_WEIGHTS does not match the expected checksum"
    cp "$DPVO_WEIGHTS" "$DPVO/dpvo.pth"
else
    echo "==> downloading DPVO weights"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT

    for url in \
        "https://huggingface.co/vslamlab/dpvo_weights/resolve/main/models.zip" \
        "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1"
    do
        echo "    trying $url"
        rm -f "$tmp/models.zip" "$tmp/dpvo.pth"
        wget -q --show-progress -O "$tmp/models.zip" "$url" || continue
        unzip -q -j -o "$tmp/models.zip" "dpvo.pth" -d "$tmp" 2>/dev/null || continue
        check_sha256 "$tmp/dpvo.pth" || continue
        mv "$tmp/dpvo.pth" "$DPVO/dpvo.pth"
        break
    done

    if [ ! -f "$DPVO/dpvo.pth" ]; then
        echo "!!  could not fetch DPVO weights (dpvo.pth, sha256 $DPVO_PTH_SHA256)." >&2
        echo "!!  Download it by hand, then re-run:" >&2
        echo "!!      DPVO_WEIGHTS=/path/to/dpvo.pth ./install.sh" >&2
        exit 1
    fi

    rm -rf "$tmp"
    trap - EXIT
fi

echo
echo "==> done. Check it with:"
echo "    python -c 'import dpvo, torch; print(torch.cuda.is_available())'"
echo "    python video_to_trajectory.py --help"
echo
echo "Metric3D's weights download themselves on first use."
echo "For DepthPro instead, see requirements-depthpro.txt."
