#include <cuda.h>
#include <lib.hpp>


// * version management

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
