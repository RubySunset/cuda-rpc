#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>

namespace srv = fractos::service::compute::cuda;


// * event management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();
    CHECK(ctx_ptr);

    auto event_ptr = ctx_ptr->make_event(Flags).get();
    CUevent cu_event = (CUevent)event_ptr.get();
    *phEvent = cu_event;

    {
        auto events_lock = std::unique_lock(state.events_mutex);
        auto res = state.events.insert(std::make_pair(cu_event, event_ptr));
        CHECK(res.second);
    }

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuEventDestroy_v2(CUevent hEvent)
{
    auto& state = get_driver_state();

    auto events_lock = std::unique_lock(state.events_mutex);

    auto it = state.events.find(hEvent);
    if (it == state.events.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    auto event_ptr = it->second;
    event_ptr->destroy().get();

    state.events.erase(it);

    return CUDA_SUCCESS;
}
