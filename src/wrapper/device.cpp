#include <cuda.h>
#include <lib.hpp>


// * device management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGet(CUdevice* device, int  ordinal)
{
    auto& state = get_state();
    LOG(WARNING) << "TODO: must keep a map of unique devices";
    auto device_ptr = state.service->device_get(ordinal).get();
    *device = device_ptr->get_device();
    {
        auto devices_lock = std::unique_lock(state.devices_mutex);
        state.devices.push_back(device_ptr);
    }
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
