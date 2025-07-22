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
    std::shared_ptr<srv::Memory> mem_ptr;
    try {
        mem_ptr = ctx_ptr->mem_alloc(size).get();
    } catch (const srv::CudaError& e) {
        return e.cuerror;
    }

    *devPtr = mem_ptr->get_deviceptr();
    state.insert_memory(mem_ptr);

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemFree_v2(CUdeviceptr dptr)
{
    auto& state = get_driver_state();

    auto mem_ptr = state.erase_memory(dptr);
    if (not mem_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    mem_ptr->destroy().get();

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemGetInfo(size_t* free, size_t* total)
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();

    try {
        std::tie(*free, *total) = ctx_ptr->mem_get_info().get();
    } catch (const srv::CudaError& e) {
        return e.cuerror;
    }

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
          size_t size,
          CUstream stream_arg)
{
    auto& state = get_driver_state();

    std::shared_ptr<srv::Memory> mem_src, mem_dst;
    switch (src.first) {
    case A:
    case D:
        mem_src = state.get_memory(src.second);
        if (not mem_src) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        break;
    case H:
        break;
    }
    switch (dst.first) {
    case A:
    case D:
        mem_dst = state.get_memory(dst.second);
        if (not mem_dst) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        break;
    case H:
        break;
    }

    if (stream_arg) {
        auto stream_ptr = state.get_stream(stream_arg);
        if (not stream_ptr) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        auto ctx_ptr = stream_ptr->get_context();
        if (not ctx_ptr) {
            return CUDA_ERROR_INVALID_HANDLE;
        }

        stream_ptr->synchronize().get();
        LOG_EVERY_N(WARNING, 100) << "TODO: enqueue cuMemcpyAync to remote service";
    } else {
        auto ctx_ptr = state.get_current_context();
        CHECK(ctx_ptr);

        ctx_ptr->synchronize().get();
    }

    if (mem_src and mem_dst) {
        // D2D
        auto& ch = get_channel();
        auto& mem_src_cap = mem_src->get_cap_mem();
        if (mem_src_cap.get_size() < size) {
            return CUDA_ERROR_INVALID_VALUE;
        } else if (mem_src_cap.get_size() > size) {
            auto src_cap = ch.diminish(mem_src_cap, 0, size, core::cap::PERM_NONE).get();
            ch.copy(src_cap, mem_dst->get_cap_mem()).get();
        } else {
            ch.copy(mem_src_cap, mem_dst->get_cap_mem()).get();
        }
        return CUDA_SUCCESS;

    } else if (mem_src and not mem_dst) {
        // D2H
        auto& ch = get_channel();
        auto dst_cap = ch.make_memory((void*)dst.second, size).get();
        auto& mem_src_cap = mem_src->get_cap_mem();
        if (mem_src_cap.get_size() < size) {
            return CUDA_ERROR_INVALID_VALUE;
        } else if (mem_src_cap.get_size() > size) {
            auto src_cap = ch.diminish(mem_src_cap, 0, size, core::cap::PERM_NONE).get();
            ch.copy(src_cap, dst_cap).get();
        } else {
            ch.copy(mem_src_cap, dst_cap).get();
        }
        return CUDA_SUCCESS;

    } else if (not mem_src and mem_dst) {
        // H2D
        auto& ch = get_channel();
        auto src_cap = ch.make_memory((const void*)src.second, size).get();
        ch.copy(src_cap, mem_dst->get_cap_mem()).get();
        LOG_EVERY_N(WARNING, 100) << "TODO: should not wait for H2D copy to finish";
        return CUDA_SUCCESS;

    } else {
        // H2H
        DCHECK(not mem_src and not mem_dst);
        memcpy((void*)dst.second, (const void*)src.second, size);
        return CUDA_SUCCESS;
    }
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return do_memcpy(std::make_pair(A, src), std::make_pair(A, dst), ByteCount, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(A, src), std::make_pair(A, dst), ByteCount, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(D, dstDevice), ByteCount, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(D, dstDevice), ByteCount, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(H, (CUdeviceptr)dstHost), ByteCount, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(H, (CUdeviceptr)dstHost), ByteCount, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    return do_memcpy(std::make_pair(H, (CUdeviceptr)srcHost), std::make_pair(D, dstDevice), ByteCount, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(H, (CUdeviceptr)srcHost), std::make_pair(D, dstDevice), ByteCount, hStream);
}


static
CUresult
do_memset(CUdeviceptr addr,
          uint64_t row_elems, uint64_t row_pad, uint64_t row_count,
          uint64_t value, uint8_t value_bytes,
          CUstream stream_arg)
{
    auto& state = get_driver_state();

    std::shared_ptr<service::compute::cuda::Context> ctx_ptr;
    std::shared_ptr<service::compute::cuda::Stream> stream_ptr;
    if (stream_arg) {
        stream_ptr = state.get_stream(stream_arg);
        if (not stream_ptr) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        ctx_ptr = stream_ptr->get_context();
        if (not ctx_ptr) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
    } else {
        ctx_ptr = state.get_current_context();
        CHECK(ctx_ptr);
    }

    try {
        switch (value_bytes) {
        case 1:
            if (stream_ptr) {
                ctx_ptr->memset(addr, row_pad, (uint8_t)(value & 0xff), row_elems, row_count, *stream_ptr).get();
            } else {
                ctx_ptr->memset(addr, row_pad, (uint8_t)(value & 0xff), row_elems, row_count).get();
            }
            break;
        case 2:
            if (stream_ptr) {
                ctx_ptr->memset(addr, row_pad, (uint16_t)(value & 0xffff), row_elems, row_count, *stream_ptr).get();
            } else {
                ctx_ptr->memset(addr, row_pad, (uint16_t)(value & 0xffff), row_elems, row_count).get();
            }
            break;
        case 4:
            if (stream_ptr) {
                ctx_ptr->memset(addr, row_pad, (uint32_t)(value & 0xffffffff), row_elems, row_count, *stream_ptr).get();
            } else {
                ctx_ptr->memset(addr, row_pad, (uint32_t)(value & 0xffffffff), row_elems, row_count).get();
            }
            break;
        default:
            LOG(FATAL) << "unexpected value_bytes=" << value_bytes;
        }
    } catch (const srv::CudaError& e) {
        return e.cuerror;
    }

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N)
{
    return do_memset(dstDevice, N, 0, 0, uc, 1, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N)
{
    return do_memset(dstDevice, N, 0, 0, us, 2, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
{
    return do_memset(dstDevice, N, 0, 0, ui, 4, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream)
{
    return do_memset(dstDevice, N, 0, 0, uc, 1, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream)
{
    return do_memset(dstDevice, N, 0, 0, us, 2, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream)
{
    return do_memset(dstDevice, N, 0, 0, ui, 4, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height)
{
    return do_memset(dstDevice, Width, dstPitch, Height, uc, 1, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height)
{
    return do_memset(dstDevice, Width, dstPitch, Height, us, 2, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height)
{
    return do_memset(dstDevice, Width, dstPitch, Height, ui, 4, 0);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream)
{
    return do_memset(dstDevice, Width, dstPitch, Height, uc, 1, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream)
{
    return do_memset(dstDevice, Width, dstPitch, Height, us, 2, hStream);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream)
{
    return do_memset(dstDevice, Width, dstPitch, Height, ui, 4, hStream);
}
