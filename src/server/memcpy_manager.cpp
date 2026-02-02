#include "./memcpy_manager.hpp"
#include "common.hpp"


MemcpyManager::~MemcpyManager() {
    for (auto& [cucontext, map] : flag_map) {
        for (auto& [custream, mem] : map) {
            checkCudaErrors(cuMemFree(mem));
        }
    }
}

void MemcpyManager::set_channel(std::shared_ptr<fractos::core::channel> ch) {
    this->ch = ch;
}

void MemcpyManager::run() {
    std::unique_lock<std::mutex> memcpy_buf_lock{memcpy_buf_mutex};

    while (!stop_flag.load()) {
        // Wait for a memcpy to be ready
        memcpy_buf_cv.wait(memcpy_buf_lock);
        std::unique_lock<std::mutex> memcpy_info_lock{memcpy_info_mutex};
        // Process all ready memcpy
        for (unsigned long memcpy_id : memcpy_buf) {
            MemcpyInfo memcpy_info = memcpy_info_map[memcpy_id];
            auto ctx = memcpy_info.ctx;
            auto stream = memcpy_info.stream;
            // TODO [ra2520] improve error handling
            auto flag = flag_map[ctx][stream];
            auto aux_stream = get_aux_stream(ctx);
            ch->copy(*memcpy_info.src, *memcpy_info.dst).then([memcpy_id, ctx, flag, aux_stream](auto& fut){
                fut.get();
                checkCudaErrors(cuCtxSetCurrent(ctx));
                // Write flag to unblock stream once memcpy is done
                checkCudaErrors(cuStreamWriteValue32(aux_stream, flag, memcpy_id, 0));
            }).as_callback();
            memcpy_info_map.erase(memcpy_id);
        }
        memcpy_buf.clear();
    }
}

void MemcpyManager::stop() {
    stop_flag.store(true);
    memcpy_buf_cv.notify_all();
}

void MemcpyManager::enqueue_memcpy_async(CUcontext ctx, CUstream stream, std::shared_ptr<fractos::core::cap::memory> src, std::shared_ptr<fractos::core::cap::memory> dst) {
    std::unique_lock<std::mutex> memcpy_info_lock{memcpy_info_mutex};
    unsigned long curr_memcpy_id = next_memcpy_id++;

    // Create a flag for this stream if one doesn't exist
    if (flag_map[ctx].find(stream) == flag_map[ctx].end()) {
        CUdeviceptr flag;
        checkCudaErrors(cuMemAlloc(&flag, 4));
        checkCudaErrors(cuMemsetD8(flag, 0, 4));
        flag_map[ctx][stream] = flag;
    }

    memcpy_info_map[curr_memcpy_id] = {
        src,
        dst,
        ctx,
        stream,
    };

    // Enqueue host callback on stream
    checkCudaErrors(cuCtxSetCurrent(ctx));
    checkCudaErrors(cuLaunchHostFunc(stream, notify_memcpy_ready, reinterpret_cast<void*>(curr_memcpy_id)));

    // Enqueue wait value on stream to block until memcpy is done
    checkCudaErrors(cuStreamWaitValue32(stream, flag_map[ctx][stream], curr_memcpy_id, CU_STREAM_WAIT_VALUE_EQ));
}

CUstream MemcpyManager::get_aux_stream(CUcontext ctx) {
    if (aux_stream_map.find(ctx) == aux_stream_map.end()) {
        checkCudaErrors(cuCtxSetCurrent(ctx));
        CUstream stream;
        checkCudaErrors(cuStreamCreate(&stream, 1));
        aux_stream_map[ctx] = stream;
    }
    return aux_stream_map[ctx];
}

void CUDA_CB notify_memcpy_ready(void* userData) {
    MemcpyManager& man = get_memcpy_manager();
    man.memcpy_buf_mutex.lock();
    man.memcpy_buf.push_back(reinterpret_cast<unsigned long>(userData));
    man.memcpy_buf_mutex.unlock();
    man.memcpy_buf_cv.notify_all();
}
