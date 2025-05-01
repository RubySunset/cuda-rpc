#include <cuda.h>
#include <fractos/service/compute/cuda.hpp>
#include <glog/logging.h>

#include "./driver-state.hpp"
#include "./driver-syms-extern.hpp"
#include "driver-state.hpp"

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


// * memory management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemAlloc(CUdeviceptr* devPtr, size_t size)
{
    auto& state = get_driver_state();

    LOG_FIRST_N(WARNING, 1) << "TODO: reserve allocated memory on host side to ensure access error?";

    auto ctx_ptr = state.get_current_context();
    auto mem_ptr = ctx_ptr->make_memory(size).get();
    *devPtr = (CUdeviceptr)mem_ptr->get_addr();

    {
        auto mems_lock = std::unique_lock(state.mems_mutex);
        auto res = state.mems.insert(std::make_pair(*devPtr, mem_ptr));
        CHECK(res.second);
    }

    return CUDA_SUCCESS;
}
