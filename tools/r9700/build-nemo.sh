#!/bin/bash
# tools/r9700/build-nemo.sh <ROCm 10 SDK dir> [build dir] [name]: strata-nemo, the Nemotron 3.5 Bolt engine, for the
# Radeon AI PRO R9700 / RX 9070 (gfx1201) against AMD TheRock ROCm 10 (its hipBLASLt 1.5 matches
# tools/hip/gfx1201-hipblaslt-100500.txt), copied to engine/<name> (default strata-nemo).
# The SDK dir is the _rocm_sdk_devel folder of a `pip install "rocm[libraries,devel,device-gfx1201]"` from
# https://nightly.repo.amd.com/rocm/whl-next/ (for example 10.2.0a20261009).
set -e
SDK=${1:?the ROCm 10 SDK dir (.../_rocm_sdk_devel)}; B=${2:-build-nemo}; NAME=${3:-strata-nemo}
cd "$(dirname "$0")/../.."
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DGPU_TARGETS=gfx1201 -DHIP_PLATFORM=amd -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=-isystem $SDK/include --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode" \
  "-DCMAKE_CXX_FLAGS=-isystem $SDK/include"
cmake --build "$B" --target strata-nemo -j "$(nproc)"
mkdir -p engine && cp "$B/strata-nemo" "engine/$NAME"
echo "built engine/$NAME"
