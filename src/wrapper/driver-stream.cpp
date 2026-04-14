#include <cuda.h>

#include <./driver-state.hpp>

namespace srv = fractos::service::compute::cuda;


// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamCreate(CUstream* phStream, unsigned int flags)
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();
    if (not ctx_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    CUresult error = CUDA_SUCCESS;
    std::shared_ptr<srv::Stream> stream;
    try {
        stream = ctx_ptr->stream_create((CUstream_flags)flags)
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    if (stream) {
        {
            auto streams_lock = std::unique_lock(state.streams_mutex);
            auto res = state.streams.insert({stream->get_stream(), stream});
            CHECK(res.second);
        }

        *phStream = stream->get_stream();
    }

    return error;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamDestroy(CUstream stream)
{
    auto& state = get_driver_state();

    auto stream_ptr = state.get_stream(stream);

    if (not stream_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        stream_ptr->destroy()
            .get();
        {
            auto streams_lock = std::unique_lock(state.streams_mutex);
            CHECK(state.streams.erase(stream) == 1);
        }
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamSynchronize(CUstream stream)
{
    auto& state = get_driver_state();

    auto stream_ptr = state.get_stream(stream);

    if (not stream_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        stream_ptr->synchronize()
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags)
{
    auto& state = get_driver_state();

    auto stream_ptr = state.get_stream(stream);
    if (not stream_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    auto event_ptr = state.get_event(event);
    if (not event_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        stream_ptr->wait_event(*event_ptr, (CUevent_wait_flags)flags)
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}
