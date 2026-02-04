#pragma once

#include <fractos/service/compute/cuda.hpp>
#include <functional>

namespace fractos::service::compute::cuda {

    namespace detail {

        template<class... Args>
        static inline
        void
        fill_args_ptr(Function& func, const void** args_ptr, const Args&... args)
        {
            std::vector<size_t> args_size;
            args_size.resize(sizeof...(Args));

            size_t index = 0;
            auto process_arg = [&](const auto& arg) {
                args_size[index] = sizeof(arg);
                args_ptr[index] = &arg;
                index++;
            };
            (process_arg(args), ...);

            func._launch_check_args(args_size);
        }

        template<class... Args>
        static inline
        std::vector<size_t>
        fill_args_ptr_and_size(const void** args_ptr, const Args&... args)
        {
            std::vector<size_t> args_size;
            args_size.resize(sizeof...(Args));

            size_t index = 0;
            auto process_arg = [&](const auto& arg) {
                args_size[index] = sizeof(arg);
                args_ptr[index] = &arg;
                index++;
            };
            (process_arg(args), ...);
            return args_size;
        }

    }

    template<class... Args>
    core::future<void>
    Function::launch(dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            detail::fill_args_ptr(*this, args_ptr, std::forward<Args>(args_)...);
        }
        return launch(args_ptr, gridDim, blockDim, 0, {});
    }

    template<class... Args>
    core::future<void>
    Function::launch(size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            detail::fill_args_ptr(*this, args_ptr, std::forward<Args>(args_)...);
        }
        return launch(args_ptr, gridDim, blockDim, sharedMem, {});
    }

    template<class... Args>
    core::future<void>
    Function::launch(Stream& stream, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            detail::fill_args_ptr(*this, args_ptr, std::forward<Args>(args_)...);
        }
        return launch(args_ptr, gridDim, blockDim, 0, stream);
    }

    template<class... Args>
    core::future<void>
    Function::launch(Stream& stream, size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args_)
    {
        const void* args_ptr[sizeof...(args_)];
        if constexpr (sizeof...(args_)) {
            detail::fill_args_ptr(*this, args_ptr, std::forward<Args>(args_)...);
        }
        return launch(args_ptr, gridDim, blockDim, sharedMem, stream);
    }

    template <class... Args>
    core::future<void>
    CublasHandle::autogen_func(uint32_t func_id, std::optional<std::reference_wrapper<Stream>> stream, Args&&... args)
    {
        const void* args_ptr[sizeof...(args)];
        std::vector<size_t> args_size;
        if constexpr (sizeof...(args)) {
            args_size = detail::fill_args_ptr_and_size(args_ptr, std::forward<Args>(args)...);
        }
        return autogen_func(args_ptr, args_size, func_id, stream);
    }
}
