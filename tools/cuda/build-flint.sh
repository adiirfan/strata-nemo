#!/bin/bash
# tools/cuda/build-flint.sh [CUDA dir] [arch]: strata-flint for an NVIDIA card, built by clang (nvcc does not take the
# fp16 vector types the kernels use) against a CUDA toolkit clang knows (12.x; default ~/cuda125, arch sm_86), into
# engine/strata-flint-cuda. The GGUF reader is header-only; no Strata library is linked, so no second
# CUDA runtime is mixed in.
set -e
CUDA=${1:-$HOME/cuda125}; ARCH=${2:-sm_86}
cd "$(dirname "$0")/../.."
B=build-flint-clang; mkdir -p "$B" engine
CXX=${CXX:-clang++}
COMMON="-O3 -std=c++20 -Iinclude -Isrc -DNDEBUG"
$CXX -x cuda --cuda-gpu-arch=$ARCH --cuda-path="$CUDA" -Wno-unknown-cuda-version $COMMON -c src/flint/flint_main.cu -o "$B/flint_main.o"
$CXX $COMMON -c src/flint/flint_cpu.cpp -o "$B/flint_cpu.o"
$CXX -o "$B/strata-flint" "$B/flint_main.o" "$B/flint_cpu.o" -L"$CUDA/lib64" -lcudart -lpthread -Wl,-rpath,"$CUDA/lib64"
rm -f engine/strata-flint-cuda && cp "$B/strata-flint" engine/strata-flint-cuda
echo "built engine/strata-flint-cuda"
