#include <cuda_runtime.h>
#include <driver_types.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>


// * event management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event, 0);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    auto& state = get_runtime_state();

    auto err = cuEventCreate(event, flags);
    return_error((cudaError_t)err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaEventDestroy(cudaEvent_t event)
{
    auto& state = get_runtime_state();

    auto err = cuEventDestroy(event);
    return_error((cudaError_t)err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaEventSynchronize(cudaEvent_t event)
{
    auto& state = get_runtime_state();

    auto err = cuEventSynchronize(event);
    return_error((cudaError_t)err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    auto& state = get_runtime_state();

    auto err = cuEventRecord(event, stream);
    return_error((cudaError_t)err);
}
