#include "./cuda_host_cb_manager.hpp"
#include "common.hpp"
#include "fractos/core/cap.hpp"
#include "fractos/service/compute/cuda_msg.hpp"
#include <cuda.h>
#include <fractos/wire/error.hpp>
#include <atomic>

#define CU_CHECK(op)                                              \
    if (CUresult cuerror = (op); cuerror != CUDA_SUCCESS) {       \
        return cuerror;                                           \
    }


CudaHostCBManager::~CudaHostCBManager()
{
    for (auto& [cucontext, map] : flag_map) {
        for (auto& [custream, mem_pair] : map) {
            checkCudaErrors(cuMemFree(mem_pair.first));
            checkCudaErrors(cuMemFree(mem_pair.second));
        }
    }
    for (auto& [cucontext, stream] : aux_stream_map) {
        checkCudaErrors(cuStreamDestroy(stream));
    }
}

void
CudaHostCBManager::set_channel(std::shared_ptr<fractos::core::channel> ch)
{
    this->ch = ch;
}

void
CudaHostCBManager::run()
{
    std::unique_lock<std::mutex> task_buf_lock{task_buf_mutex};

    while (!stop_flag.load()) {
        // Wait for a task to be ready
        task_buf_cv.wait(task_buf_lock);
        // Process ready memcpy
        for (unsigned long task_id : task_buf) {
            MemcpyInfo memcpy_info;
            {
                std::unique_lock<std::mutex> memcpy_info_lock{memcpy_info_mutex};
                auto it = memcpy_info_map.find(task_id);
                if (it == memcpy_info_map.end()) {
                    continue;
                }
                memcpy_info = it->second;
                memcpy_info_map.erase(it);
            }
            auto ctx = memcpy_info.ctx;
            auto stream = memcpy_info.stream;
            // TODO [ra2520] improve error handling
            auto memcpy_flag = flag_map[ctx][stream].second;
            auto aux_stream = get_aux_stream(ctx);
            ch->copy(*memcpy_info.src, *memcpy_info.dst).then([src=memcpy_info.src, dst=memcpy_info.dst, task_id, ctx, memcpy_flag, aux_stream](auto& fut){
                fut.get();
                cuCtxSetCurrent(ctx);
                // Write flag to unblock stream once memcpy is done
                cuStreamWriteValue32(aux_stream, memcpy_flag, task_id, 0);
            }).as_callback();
        }
        // Process ready streams
        {
            std::unique_lock<std::mutex> stream_cont_lock{stream_cont_mutex};
            for (unsigned long task_id : task_buf) {
                if (stream_cont_map.find(task_id) == stream_cont_map.end()) {
                    continue;
                }
                auto& cont = stream_cont_map[task_id];
                using msg = fractos::service::compute::cuda::wire::Stream::synchronize;
                ch->template make_request_builder<msg::response>(cont)
                    .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                    .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
                    .on_channel()
                    .invoke()
                    .as_callback();
                stream_cont_map.erase(task_id);
            }
        }
        // Process ready contexts
        {
            std::unique_lock<std::mutex> ctx_info_lock{ctx_info_mutex};
            for (unsigned long task_id : task_buf) {
                if (ctx_cont_map.find(task_id) == ctx_cont_map.end()) {
                    continue;
                }
                auto& cont = ctx_cont_map[task_id];
                using msg = fractos::service::compute::cuda::wire::Context::synchronize;
                ch->template make_request_builder<msg::response>(cont)
                    .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                    .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
                    .on_channel()
                    .invoke()
                    .as_callback();
                ctx_cont_map.erase(task_id);
            }
        }
        task_buf.clear();
    }
}

void
CudaHostCBManager::stop()
{
    stop_flag.store(true);
    task_buf_cv.notify_all();
}

CUresult
CudaHostCBManager::enqueue_memcpy_async(CUcontext ctx, CUstream stream, std::shared_ptr<fractos::core::cap::memory> src, std::shared_ptr<fractos::core::cap::memory> dst)
{
    CU_CHECK(cuCtxSetCurrent(ctx));
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);

    {
        std::unique_lock<std::mutex> memcpy_info_lock{memcpy_info_mutex};
        // Create flags for this stream if one doesn't exist
        if (flag_map[ctx].find(stream) == flag_map[ctx].end()) {
            CUdeviceptr temp_flag, memcpy_flag;
            CU_CHECK(cuMemAlloc(&temp_flag, 4));
            CU_CHECK(cuMemAlloc(&memcpy_flag, 4));
            flag_map[ctx][stream] = {temp_flag, memcpy_flag};
            CU_CHECK(cuMemsetD8(temp_flag, 0, 4));
            CU_CHECK(cuMemsetD8(memcpy_flag, 0, 4));
        }

        memcpy_info_map[curr_task_id] = {
            src,
            dst,
            ctx,
            stream,
        };
    }

    // Block stream
    CU_CHECK(cuStreamWaitValue32(stream, flag_map[ctx][stream].first, 1, CU_STREAM_WAIT_VALUE_EQ));

    // Enqueue host callback on stream
    CU_CHECK(cuLaunchHostFunc(stream, notify_task_ready, reinterpret_cast<void*>(curr_task_id)));

    // Enqueue wait value on stream to block until memcpy is done
    CU_CHECK(cuStreamWaitValue32(stream, flag_map[ctx][stream].second, curr_task_id, CU_STREAM_WAIT_VALUE_EQ));

    // Unblock stream - now CUDA host function can be called
    auto aux_stream = get_aux_stream(ctx);
    CU_CHECK(cuStreamWriteValue32(aux_stream, flag_map[ctx][stream].first, 1, 0));

    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::stream_sync(CUcontext ctx, CUstream stream, core::cap::request continuation)
{
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);
    {
        std::unique_lock<std::mutex> stream_cont_lock{stream_cont_mutex};
        stream_cont_map[curr_task_id] = std::move(continuation);
    }

    // Enqueue host callback on stream
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuLaunchHostFunc(stream, notify_task_ready, reinterpret_cast<void*>(curr_task_id)));

    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::ctx_sync(CUcontext ctx, std::vector<CUstream>& streams, core::cap::request continuation)
{
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);
    {
        std::unique_lock<std::mutex> ctx_info_lock{ctx_info_mutex};
        ctx_cont_map[curr_task_id] = std::move(continuation);
    }
    num_remaining_streams_map[curr_task_id].store(streams.size());

    // Enqueue host callback on each stream
    CU_CHECK(cuCtxSetCurrent(ctx));
    for (CUstream stream : streams) {
        CU_CHECK(cuLaunchHostFunc(stream, update_remaining_streams, reinterpret_cast<void*>(curr_task_id)));
    }
    // TODO [ra2520] if any of the host functions fail to launch, the context sync will never be completed
    // since the count of completed streams will be less than required. This is fine for correctness but
    // does leave some stale data that isn't cleaned up properly.

    return CUDA_SUCCESS;
}

CUstream
CudaHostCBManager::get_aux_stream(CUcontext ctx)
{
    if (aux_stream_map.find(ctx) == aux_stream_map.end()) {
        cuCtxSetCurrent(ctx);
        CUstream stream;
        cuStreamCreate(&stream, 1);
        aux_stream_map[ctx] = stream;
    }
    return aux_stream_map[ctx];
}

void
CUDA_CB
notify_task_ready(void* userData)
{
    CudaHostCBManager& man = get_cuda_host_cb_manager();
    man.task_buf_mutex.lock();
    man.task_buf.push_back(reinterpret_cast<unsigned long>(userData));
    man.task_buf_mutex.unlock();
    man.task_buf_cv.notify_all();
}

void
CUDA_CB
update_remaining_streams(void* userData)
{
    CudaHostCBManager& man = get_cuda_host_cb_manager();
    auto task_id = reinterpret_cast<unsigned long>(userData);
    if (man.num_remaining_streams_map[task_id].fetch_sub(1) == 1) {
        man.num_remaining_streams_map.erase(task_id);
        man.task_buf_mutex.lock();
        man.task_buf.push_back(task_id);
        man.task_buf_mutex.unlock();
        man.task_buf_cv.notify_all();
    }
}
