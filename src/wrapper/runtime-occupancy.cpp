#include <cuda.h>
#include <cuda_runtime_api.h>

#include "./runtime-state.hpp"


// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize)
{
    return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, CU_OCCUPANCY_DEFAULT);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    auto& state = get_runtime_state();

    auto [err, cufunc] = state.get_function(func);
    return_error_maybe(err);

    auto cuerror = (cudaError)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, cufunc, blockSize, dynamicSMemSize, flags);
    return_error(cuerror);
}
