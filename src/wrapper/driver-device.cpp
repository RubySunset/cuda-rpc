#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>


// * device management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGet(CUdevice* device, int  ordinal)
{
    auto& state = get_driver_state();
    auto device_ptr = state.get_device_ordinal(ordinal);
    if (not device_ptr) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = device_ptr->get_device();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    auto& state = get_driver_state();
    auto device = state.get_device(dev);
    if (not device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *pi = device->get_attribute(attrib).get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetCount(int* count)
{
    auto& state = get_driver_state();
    *count = state.service->device_get_count().get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetName(char* name, int  len, CUdevice dev)
{
    auto& state = get_driver_state();
    auto device = state.get_device(dev);
    if (not device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    auto name_str = device->get_name().get();
    strncat(name, name_str.c_str(), len);
    name[len-1] = 0;
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev)
{
    auto& state = get_driver_state();
    auto device = state.get_device(dev);
    if (not device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *uuid = device->get_uuid().get();
    return CUDA_SUCCESS;
}


extern "C" [[gnu::visibility("default")]]
CUresult
cuDeviceTotalMem_v2(size_t* bytes, CUdevice dev)
{
    auto& state = get_driver_state();
    auto device = state.get_device(dev);
    if (not device) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = device->total_mem().get();
    return CUDA_SUCCESS;
}

