#include <dlfcn.h>
#include <cuda.h>

#include <fractos/common/logging.hpp>
#include <./common.hpp>
#include <./driver-state.hpp>
#include <driver-lib.hpp>


// * initialization
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuInit(unsigned int flags)
{
    auto lock = std::unique_lock(get_driver_state_mutex());

    if (get_driver_state_atomic().load()) {
        return CUDA_SUCCESS;
    }

    auto ch = get_channel_ptr();
    auto gns = fractos::core::gns::make_service();

    auto name = get_env("FRACTOS_SERVICE_COMPUTE_CUDA_NAME",
                        "fractos::service::compute::cuda");

    fractos::common::logging::init(const_cast<char*>(name.c_str()));
    LOG(INFO) << "Initialized logging";

    auto state = std::make_shared<DriverState>();

    state->service = fractos::service::compute::cuda::make_service(ch, *gns, name).get();
    state->service->init(flags).get();

    if (state->service->device_get_count().get() == 0) {
        return CUDA_ERROR_NO_DEVICE;
    }

    auto prev = get_driver_state_atomic().exchange(state);
    CHECK(not prev);

    return CUDA_SUCCESS;
}
