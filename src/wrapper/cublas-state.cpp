#include "./cublas-state.hpp"


void
CublasState::add_handle(cublasHandle_t handle, std::shared_ptr<srv::CublasHandle> cublas_obj)
{
    std::unique_lock lock(mut);
    cublas_handle_map[handle] = cublas_obj;
}

std::shared_ptr<srv::CublasHandle>
CublasState::get_handle(cublasHandle_t handle)
{
    std::unique_lock lock(mut);
    auto it = cublas_handle_map.find(handle);
    if (it == cublas_handle_map.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

bool
CublasState::erase_handle(cublasHandle_t handle)
{
    std::unique_lock lock(mut);
    return cublas_handle_map.erase(handle);
}

void
CublasState::update_stream(cublasHandle_t handle, std::shared_ptr<srv::Stream> stream_obj)
{
    std::unique_lock lock(mut);
    stream_map[handle] = stream_obj;
}

std::shared_ptr<srv::Stream>
CublasState::get_stream(cublasHandle_t handle)
{
    std::unique_lock lock(mut);
    auto it = stream_map.find(handle);
    if (it == stream_map.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

bool
CublasState::erase_stream(cublasHandle_t handle)
{
    std::unique_lock lock(mut);
    return stream_map.erase(handle);
}


CublasState _cublas_state;
