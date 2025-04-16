#include <cuda.h>
#include <lib.hpp>


// * device management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetCount(int* count)
{
    auto& state = get_state();
    *count = state.service->device_get_count().get();
    return CUDA_SUCCESS;
}
