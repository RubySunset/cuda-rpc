#pragma once

#include <fractos/service/compute/cuda.hpp>

namespace fractos::service::compute::cuda {

    namespace detail {

        template<class... Args>
        void
        fill_args_ptr(Function& func, const void** args_ptr, std::tuple<Args...>& args)
        {
            size_t i = 0;
            std::vector<size_t> args_size;
            args_size.resize(sizeof...(Args));
            std::apply([&](auto& arg) {
                args_size.push_back(sizeof(arg));
                args_ptr[i] = &arg;
                i++;
            }, args);
            func._launch_check_args(args_size);
        }

    }

    template<class... Args>
    core::future<void>
    Function::launch(dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            std::tuple<Args...> args(std::forward<Args>(args_)...);
            detail::fill_args_ptr(*this, args_ptr, args);
        }
        return launch(args_ptr, gridDim, blockDim, 0, {});
    }

    template<class... Args>
    core::future<void>
    Function::launch(size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            std::tuple<Args...> args(std::forward<Args>(args_)...);
            detail::fill_args_ptr(*this, args_ptr, args);
        }
        return launch(args_ptr, gridDim, blockDim, sharedMem, {});
    }

    template<class... Args>
    core::future<void>
    Function::launch(Stream& stream, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            std::tuple<Args...> args(std::forward<Args>(args_)...);
            detail::fill_args_ptr(*this, args_ptr, args);
        }
        return launch(args_ptr, gridDim, blockDim, 0, stream);
    }

    template<class... Args>
    core::future<void>
    Function::launch(Stream& stream, size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            std::tuple<Args...> args(std::forward<Args>(args_)...);
            detail::fill_args_ptr(*this, args_ptr, args);
        }
        return launch(args_ptr, gridDim, blockDim, sharedMem, stream);
    }

}
