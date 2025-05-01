#include "runtime-state.hpp"
#include <cuda_runtime.h>

#include "./runtime-state.hpp"
#include "./runtime-syms-extern.hpp"


// * memory management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMalloc(void** devPtr, size_t size)
{
    auto& state = get_runtime_state();
    auto err = (cudaError_t)cuMemAlloc((CUdeviceptr*)devPtr, size);
    return_error(err);
}
