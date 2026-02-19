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
          CUstream stream_arg,
          bool is_sync)
{
    auto& state = get_driver_state();

    auto invalid_mem_kind = [&](std::shared_ptr<srv::Memory>& mem, pointer_type kind) {
        return (mem and kind == H) || (not mem and kind == D);
    };

    auto mem_src = state.get_memory(src.second);
    auto mem_dst = state.get_memory(dst.second);
    if (invalid_mem_kind(mem_src, src.first) or invalid_mem_kind(mem_dst, dst.first)) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto ctx_ptr = state.get_current_context();
    if (not ctx_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    auto stream_ptr = state.get_stream(stream_arg);
    if (not stream_ptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    auto capture_mem_until_sync = [&](auto& vec){
        stream_ptr->synchronize()
        .then([moved_vec=std::move(vec)](auto& fut){
            fut.get();
        })
        .as_callback();
    };

    std::vector<core::cap::memory> captured_mem;

    // See below for sync/async memcpy semantics:
    // https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html

    if (mem_src and mem_dst) {
        // D2D
        // Is always async, regardless of which API is used
        auto& ch = get_channel();
        auto& mem_src_cap = mem_src->get_cap_mem();
        if (mem_src->get_size() < size) {
            return CUDA_ERROR_INVALID_VALUE;
        } else if (mem_src->get_size() > size) {
            auto src_cap = ch.diminish(mem_src_cap, 0, size, core::cap::PERM_NONE).get();
            ctx_ptr->memcpy_async(src_cap, mem_dst->get_cap_mem(), *stream_ptr).get();
            captured_mem.push_back(std::move(src_cap));
            capture_mem_until_sync(captured_mem);
        } else {
            ctx_ptr->memcpy_async(mem_src_cap, mem_dst->get_cap_mem(), *stream_ptr).get();
        }
        return CUDA_SUCCESS;

    } else if (mem_src and not mem_dst) {
        // D2H
        auto& ch = get_channel();
        auto dst_cap = ch.make_memory((void*)dst.second, size).get();
        auto& mem_src_cap = mem_src->get_cap_mem();
        if (mem_src->get_size() < size) {
            return CUDA_ERROR_INVALID_VALUE;
        } else if (mem_src->get_size() > size) {
            auto src_cap = ch.diminish(mem_src_cap, 0, size, core::cap::PERM_NONE).get();
            ctx_ptr->memcpy_async(src_cap, dst_cap, *stream_ptr).get();
            captured_mem.push_back(std::move(src_cap));
        } else {
            ctx_ptr->memcpy_async(mem_src_cap, dst_cap, *stream_ptr).get();
        }
        if (is_sync) {
            return cuStreamSynchronize(stream_arg);
        } else {
            captured_mem.push_back(std::move(dst_cap));
            capture_mem_until_sync(captured_mem);
            return CUDA_SUCCESS;
        }

    } else if (not mem_src and mem_dst) {
        // H2D
        auto& ch = get_channel();
        auto src_cap = ch.make_memory((const void*)src.second, size).get();
        ctx_ptr->memcpy_async(src_cap, mem_dst->get_cap_mem(), *stream_ptr).get();
        if (is_sync) {
            return cuStreamSynchronize(stream_arg);
        } else {
            captured_mem.push_back(std::move(src_cap));
            capture_mem_until_sync(captured_mem);
            return CUDA_SUCCESS;
        }

    } else {
        // H2H
        // Is always sync, regardless of which API is used
        DCHECK(not mem_src and not mem_dst);
        CUresult cuerror = cuStreamSynchronize(stream_arg);
        if (cuerror != CUDA_SUCCESS) {
            return cuerror;
        }
        memcpy((void*)dst.second, (const void*)src.second, size);
        return CUDA_SUCCESS;
    }
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount)
{
    return do_memcpy(std::make_pair(A, src), std::make_pair(A, dst), ByteCount, 0, true);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(A, src), std::make_pair(A, dst), ByteCount, hStream, false);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(D, dstDevice), ByteCount, 0, true);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(D, dstDevice), ByteCount, hStream, false);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoH_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(H, (CUdeviceptr)dstHost), ByteCount, 0, true);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyDtoHAsync_v2(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(D, srcDevice), std::make_pair(H, (CUdeviceptr)dstHost), ByteCount, hStream, false);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount)
{
    return do_memcpy(std::make_pair(H, (CUdeviceptr)srcHost), std::make_pair(D, dstDevice), ByteCount, 0, true);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream)
{
    return do_memcpy(std::make_pair(H, (CUdeviceptr)srcHost), std::make_pair(D, dstDevice), ByteCount, hStream, false);
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
    std::shared_ptr<service::compute::cuda::Stream> stream_ptr = state.get_stream(stream_arg);

    if (not stream_ptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    ctx_ptr = stream_ptr->get_context();
    if (not ctx_ptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    try {
        switch (value_bytes) {
        case 1:
            ctx_ptr->memset(addr, row_pad, (uint8_t)(value & 0xff), row_elems, row_count, *stream_ptr).get();
            break;
        case 2:
            ctx_ptr->memset(addr, row_pad, (uint16_t)(value & 0xffff), row_elems, row_count, *stream_ptr).get();
            break;
        case 4:
            ctx_ptr->memset(addr, row_pad, (uint32_t)(value & 0xffffffff), row_elems, row_count, *stream_ptr).get();
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
