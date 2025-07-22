#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>
#include <./runtime-syms-extern.hpp>


// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaStreamCreate(cudaStream_t* pStream)
{
    return cudaStreamCreateWithFlags(pStream, cudaStreamDefault);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags)
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuStreamCreate(pStream, flags);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaStreamDestroy(cudaStream_t stream)
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuStreamDestroy(stream);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaStreamSynchronize(cudaStream_t stream)
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuStreamSynchronize(stream);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuStreamWaitEvent(stream, event, flags);
    return_error(err);
}
