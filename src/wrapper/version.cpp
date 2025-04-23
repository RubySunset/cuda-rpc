#include <cuda.h>

#include <./state.hpp>


// * version management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION

extern "C" [[gnu::visibility("default")]]
CUresult
cuDriverGetVersion(int* driverVersion)
{
    if (not driverVersion) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto& state = get_state();
    *driverVersion = state.service->get_driver_version().get();
    return CUDA_SUCCESS;
}
