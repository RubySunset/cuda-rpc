#include <cuda_runtime.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>
#include <./runtime-syms-extern.hpp>


// * execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            struct CUstream_st *stream)
{
    return (*ptr___cudaPushCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                           void *stream)
{
    return (*ptr___cudaPopCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}
