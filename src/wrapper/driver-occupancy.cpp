#include <cuda.h>

#include "./driver-state.hpp"


namespace clt = fractos::service::compute::cuda;


// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize)
{
    return cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks, func, blockSize, dynamicSMemSize, CU_OCCUPANCY_DEFAULT);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int flags)
{
    auto& state = get_driver_state();

    auto func_desc = state.get_function(func);
    if (not func_desc) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        *numBlocks = func_desc->function->occupancy_max_active_blocks_per_multiprocessor_with_flags(
            blockSize, dynamicSMemSize, (CUoccupancy_flags)flags)
            .get();
    } catch (const clt::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}
