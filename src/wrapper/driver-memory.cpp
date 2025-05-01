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

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemFree_v2(CUdeviceptr dptr)
{
    auto& state = get_driver_state();

    std::shared_ptr<srv::Memory> mem_ptr;
    {
        auto mems_lock = std::unique_lock(state.mems_mutex);
        auto it = state.mems.find(dptr);
        if (it == state.mems.end()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        mem_ptr = it->second;
        state.mems.erase(it);
    }

    mem_ptr->destroy().get();

    return CUDA_SUCCESS;
}

enum pointer_type {
    A,
    H,
    D,
};

static
CUresult
do_memcpy(std::pair<pointer_type, CUdeviceptr> src,
          std::pair<pointer_type, CUdeviceptr> dst,
          size_t size)
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();
    CHECK(ctx_ptr);

    std::shared_ptr<srv::Memory> mem_src, mem_dst;
    {
        auto mems_lock = std::shared_lock(state.mems_mutex);
        switch (src.first) {
        case A:
        case D:
        {
            auto it = state.mems.find(src.second);
            if (it != state.mems.end()) {
                mem_src = it->second;
            }
            break;
        }
        case H:
            break;
        }
        switch (dst.first) {
        case A:
        case D:
        {
            auto it = state.mems.find(dst.second);
            if (it != state.mems.end()) {
                mem_dst = it->second;
            }
            break;
        }
        case H:
            break;
        }
    }

    if (mem_src and mem_dst) {
        // D2D
        auto& ch = get_channel();
        ch.copy(mem_src->get_cap_mem(), mem_dst->get_cap_mem()).get();
        return CUDA_SUCCESS;

    } else if (mem_src and not mem_dst) {
        // D2H
        ctx_ptr->synchronize().get();
        auto& ch = get_channel();
        auto cap_dst = ch.make_memory((void*)dst.second, size).get();
        ch.copy(mem_src->get_cap_mem(), cap_dst).get();
        return CUDA_SUCCESS;

    } else if (not mem_src and mem_dst) {
        // H2D
        ctx_ptr->synchronize().get();
        auto& ch = get_channel();
        auto cap_src = ch.make_memory((const void*)src.second, size).get();
        ch.copy(cap_src, mem_dst->get_cap_mem()).get();
        LOG_FIRST_N(WARNING, 1) << "TODO: should not wait for H2D copy to finish";
        return CUDA_SUCCESS;

    } else {
        // H2H
        DCHECK(not mem_src and not mem_dst);
        ctx_ptr->synchronize().get();
        memcpy((void*)dst.second, (const void*)src.second, size);
        return CUDA_SUCCESS;
    }
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return do_memcpy(std::make_pair(A, src), std::make_pair(A, dst), ByteCount);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(D, dstDevice), ByteCount);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(H, (CUdeviceptr)dstHost), ByteCount);
}
