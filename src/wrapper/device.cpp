#include <cuda.h>
#include <lib.hpp>


// * device management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGet(CUdevice* device, int  ordinal)
{
    auto& state = get_state();
    auto device_ptr = state.get_device_ordinal(ordinal);
    *device = device_ptr->get_device();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetCount(int* count)
{
    auto& state = get_state();
    *count = state.service->device_get_count().get();
    return CUDA_SUCCESS;
}
