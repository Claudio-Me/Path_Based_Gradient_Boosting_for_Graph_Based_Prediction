#!/bin/bash
#
# Compile the C++ graph-kernel baselines into a Python extension.
#
# Needed only for the WL, Graphlet, Shortest-path and WLOA baselines
# (run_kernel.py). PathBoost and the GNN baseline do not use it.
#
# Requirements:
#   g++ (or clang++) with C++17 (eigen 5 requires at least C++14)
#   eigen3 headers   -- apt install libeigen3-dev  /  brew install eigen
#   pybind11         -- installed by requirements.txt
#
# Usage:
#   ./build_kernels.sh                 # uses `python` from the active venv
#   PYTHON=python3.12 ./build_kernels.sh
#   EIGEN_FLAGS=-I/path/to/eigen3 ./build_kernels.sh

set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

command -v "$PYTHON" >/dev/null || { echo "error: '$PYTHON' not found. Activate your virtual environment or set PYTHON=..." >&2; exit 1; }
# Resolve to an absolute path: the build runs from a subdirectory, where a
# relative interpreter path such as venv/bin/python would no longer exist.
PYTHON="$(command -v "$PYTHON")"
case "$PYTHON" in /*) ;; *) PYTHON="$PWD/$PYTHON" ;; esac
command -v g++      >/dev/null || { echo "error: g++ not found. Install a C++ compiler." >&2; exit 1; }

"$PYTHON" -c "import pybind11" 2>/dev/null || {
    echo "error: pybind11 is not installed in $PYTHON. Run: pip install -r requirements.txt" >&2
    exit 1
}

# Locate the eigen3 headers. They are not on the default include path on macOS
# and live in an eigen3/ subdirectory on most Linux distributions.
EIGEN_FLAGS="${EIGEN_FLAGS:-}"
if [ -n "$EIGEN_FLAGS" ]; then
    :   # honour an explicit EIGEN_FLAGS from the environment
elif pkg-config --exists eigen3 2>/dev/null; then
    EIGEN_FLAGS="$(pkg-config --cflags eigen3)"
else
    for candidate in /opt/homebrew/include/eigen3 /usr/local/include/eigen3 /usr/include/eigen3; do
        if [ -d "$candidate/Eigen" ]; then
            EIGEN_FLAGS="-I$candidate"
            break
        fi
    done
fi

if [ -z "$EIGEN_FLAGS" ]; then
    echo "error: could not find the eigen3 headers." >&2
    echo "  Debian/Ubuntu: sudo apt install libeigen3-dev" >&2
    echo "  macOS:         brew install eigen" >&2
    echo "  Already installed elsewhere? Re-run with EIGEN_FLAGS=-I/path/to/eigen3" >&2
    exit 1
fi
echo "eigen3: $EIGEN_FLAGS"

# Linux needs -fPIC; macOS needs to leave the Python symbols undefined.
case "$(uname -s)" in
    Darwin) PLATFORM_FLAGS="-undefined dynamic_lookup" ;;
    *)      PLATFORM_FLAGS="-fPIC" ;;
esac

SUFFIX="$("$PYTHON"-config --extension-suffix 2>/dev/null || "$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUTPUT="../kernel_baselines${SUFFIX}"

cd tudataset/tud_benchmark/kernel_baselines
echo "building -> tudataset/tud_benchmark/kernel_baselines${SUFFIX}"
# shellcheck disable=SC2086
g++ -O3 -shared -std=c++17 $PLATFORM_FLAGS $EIGEN_FLAGS \
    $("$PYTHON" -m pybind11 --includes) \
    kernel_baselines.cpp src/*.cpp -o "$OUTPUT"

cd ../../..
echo "verifying..."
"$PYTHON" -c "import tudataset.tud_benchmark.kernel_baselines as kb; print('OK:', kb.compute_wl_1_dense)"
