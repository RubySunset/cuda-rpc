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

// uncomment the following line to force operations like cuCtxSynchronize
// to block the cuda service, rather than using host callbacks to be
// fully asynchronous

// #define FORCE_SYNC


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
CudaHostCBManager::process_ready_memcpy(unsigned int task_id)
{
    MemcpyInfo memcpy_info;
    {
        std::unique_lock<std::mutex> memcpy_info_lock{memcpy_info_mutex};
        auto it = memcpy_info_map.find(task_id);
        if (it == memcpy_info_map.end()) {
            LOG(WARNING) << "Task " << task_id << " has task type memcpy, but no matching entry exists in memcpy_info";
            return;
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

void
CudaHostCBManager::process_ready_stream(unsigned int task_id)
{
    std::unique_lock<std::mutex> stream_cont_lock{stream_cont_mutex};
    auto it = stream_cont_map.find(task_id);
    if (it == stream_cont_map.end()) {
        LOG(WARNING) << "Task " << task_id << " has task type stream sync, but no matching entry exists in stream_cont_map";
        return;
    }
    using msg = fractos::service::compute::cuda::wire::Stream::synchronize;
    ch->template make_request_builder<msg::response>(it->second)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
    stream_cont_map.erase(it);
}

void
CudaHostCBManager::process_ready_ctx(unsigned int task_id)
{
    std::unique_lock<std::mutex> ctx_info_lock{ctx_info_mutex};
    auto it = ctx_cont_map.find(task_id);
    if (it == ctx_cont_map.end()) {
        LOG(WARNING) << "Task " << task_id << " has task type ctx sync, but no matching entry exists in ctx_cont_map";
        return;
    }
    using msg = fractos::service::compute::cuda::wire::Context::synchronize;
    ch->template make_request_builder<msg::response>(it->second)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
    ctx_cont_map.erase(it);
}

void
CudaHostCBManager::process_ready_event(unsigned int task_id)
{
    std::unique_lock<std::mutex> event_lock{event_mutex};
    auto it = event_info.find(task_id);
    if (it == event_info.end()) {
        LOG(WARNING) << "Task " << task_id << " has task type event sync, but no matching entry exists in event_info";
        return;
    }
    it->second.ready = true;
    if (not it->second.cont) {
        return;
    }
    using msg = fractos::service::compute::cuda::wire::Event::synchronize;
    ch->template make_request_builder<msg::response>(*it->second.cont)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
    auto cuevent = id_to_event[task_id];
    event_to_id.erase(cuevent);
    id_to_event.erase(task_id);
    event_info.erase(it);
}

void
CudaHostCBManager::run()
{
    std::unique_lock<std::mutex> task_buf_lock{task_buf_mutex};

    while (!stop_flag.load()) {
        // Wait for a task to be ready
        task_buf_cv.wait(task_buf_lock);
#ifndef FORCE_SYNC
        for (auto [task_type, task_id] : task_buf) {
            switch (task_type) {
                case TaskType::MEMCPY_ASYNC:
                process_ready_memcpy(task_id);
                break;
                case TaskType::STREAM_SYNC:
                process_ready_stream(task_id);
                break;
                case TaskType::CTX_SYNC:
                process_ready_ctx(task_id);
                break;
                case TaskType::EVENT_SYNC:
                process_ready_event(task_id);
                break;
            }
        }
        task_buf.clear();
#endif
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
#ifdef FORCE_SYNC
    CU_CHECK(cuStreamSynchronize(stream));
    ch->copy(*src, *dst).get();
#else
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
    CU_CHECK(cuLaunchHostFunc(stream, notify_memcpy_ready, reinterpret_cast<void*>(curr_task_id)));

    // Enqueue wait value on stream to block until memcpy is done
    CU_CHECK(cuStreamWaitValue32(stream, flag_map[ctx][stream].second, curr_task_id, CU_STREAM_WAIT_VALUE_EQ));

    // Unblock stream - now CUDA host function can be called
    auto aux_stream = get_aux_stream(ctx);
    CU_CHECK(cuStreamWriteValue32(aux_stream, flag_map[ctx][stream].first, 1, 0));
#endif
    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::stream_sync(CUcontext ctx, CUstream stream, core::cap::request continuation)
{
#ifdef FORCE_SYNC
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuStreamSynchronize(stream));
    using msg = fractos::service::compute::cuda::wire::Stream::synchronize;
    ch->template make_request_builder<msg::response>(continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
#else
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);
    {
        std::unique_lock<std::mutex> stream_cont_lock{stream_cont_mutex};
        stream_cont_map[curr_task_id] = std::move(continuation);
    }

    // Enqueue host callback on stream
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuLaunchHostFunc(stream, notify_stream_ready, reinterpret_cast<void*>(curr_task_id)));
#endif
    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::ctx_sync(CUcontext ctx, std::vector<CUstream>& streams, core::cap::request continuation)
{
#ifdef FORCE_SYNC
    CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuCtxSynchronize());
    using msg = fractos::service::compute::cuda::wire::Context::synchronize;
    ch->template make_request_builder<msg::response>(continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
#else
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
#endif
    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::event_create(CUevent event)
{
#ifndef FORCE_SYNC
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);
    {
        std::unique_lock<std::mutex> event_lock{event_mutex};
        event_to_id[event] = curr_task_id;
        id_to_event[curr_task_id] = event;
        event_info[curr_task_id] = {true, nullptr};
    }
#endif
    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::event_record(CUstream stream, CUevent event)
{
#ifdef FORCE_SYNC
    CU_CHECK(cuEventRecord(event, stream));
#else
    unsigned long curr_task_id = next_task_id.fetch_add(1, std::memory_order::relaxed);
    {
        std::unique_lock<std::mutex> event_lock{event_mutex};
        event_to_id[event] = curr_task_id;
        id_to_event[curr_task_id] = event;
        event_info[curr_task_id] = {false, nullptr};
    }

    // Enqueue host callback on stream
    // CU_CHECK(cuCtxSetCurrent(ctx));
    CU_CHECK(cuLaunchHostFunc(stream, notify_event_ready, reinterpret_cast<void*>(curr_task_id)));
#endif
    return CUDA_SUCCESS;
}

CUresult
CudaHostCBManager::event_sync(CUevent event, core::cap::request continuation)
{
#ifdef FORCE_SYNC
    CU_CHECK(cuEventSynchronize(event));
    using msg = fractos::service::compute::cuda::wire::Event::synchronize;
    ch->template make_request_builder<msg::response>(continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
#else
    std::unique_lock<std::mutex> event_lock{event_mutex};
    auto id = event_to_id[event];
    auto& state = event_info[id];
    if (state.ready) {
        using msg = fractos::service::compute::cuda::wire::Event::synchronize;
        ch->template make_request_builder<msg::response>(continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
            .set_imm(&msg::response::imms::cuerror, CUDA_SUCCESS)
            .on_channel()
            .invoke()
            .as_callback();
        event_to_id.erase(event);
        id_to_event.erase(id);
        event_info.erase(id);
    } else {
        state.cont = std::make_shared<core::cap::request>(std::move(continuation));
    }
#endif
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
notify_memcpy_ready(void* userData)
{
    CudaHostCBManager& man = get_cuda_host_cb_manager();
    man.task_buf_mutex.lock();
    man.task_buf.emplace_back(TaskType::MEMCPY_ASYNC, reinterpret_cast<unsigned long>(userData));
    man.task_buf_mutex.unlock();
    man.task_buf_cv.notify_all();
}

void
CUDA_CB
notify_stream_ready(void* userData)
{
    CudaHostCBManager& man = get_cuda_host_cb_manager();
    man.task_buf_mutex.lock();
    man.task_buf.emplace_back(TaskType::STREAM_SYNC, reinterpret_cast<unsigned long>(userData));
    man.task_buf_mutex.unlock();
    man.task_buf_cv.notify_all();
}

void
CUDA_CB
notify_event_ready(void* userData)
{
    CudaHostCBManager& man = get_cuda_host_cb_manager();
    man.task_buf_mutex.lock();
    man.task_buf.emplace_back(TaskType::EVENT_SYNC, reinterpret_cast<unsigned long>(userData));
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
        man.task_buf.emplace_back(TaskType::CTX_SYNC, task_id);
        man.task_buf_mutex.unlock();
        man.task_buf_cv.notify_all();
    }
}
