#!/bin/bash
# tools/flint/run.sh [engine args]: strata-flint with the ROCm 10 libraries on the path.
# ROCM_SDK: the _rocm_sdk_devel folder; FLINT_GGUF: the model; FLINT_BIN: the engine (default engine/strata-flint)
SDK=${ROCM_SDK:?set ROCM_SDK to the ROCm 10 _rocm_sdk_devel folder}
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib
exec "${FLINT_BIN:-$(dirname "$0")/../../engine/strata-flint}" --model "${FLINT_GGUF:?set FLINT_GGUF to the Flint GGUF}" "$@"
