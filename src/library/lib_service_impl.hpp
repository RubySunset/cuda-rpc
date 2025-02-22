// #pragma once

#include <fractos/service/compute/cuda.hpp>

using namespace fractos;

namespace impl {
    struct CuService_impl;
    struct CuDevice_impl;

    std::string to_string(const CuService_impl& obj);

    struct CuService_impl {
        // CuService_impl(CuService_impl& other) = delete;
        // CuService_impl(const CuService_impl& other) = delete;
        
        // pimpl-to-impl
        static inline impl::CuService_impl& get(fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<impl::CuService_impl*>(obj._pimpl.get()); }
        static inline const impl::CuService_impl& get(const fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<const impl::CuService_impl*>(obj._pimpl.get()); }

        static inline std::shared_ptr<impl::CuService_impl> get_ptr(fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<impl::CuService_impl>(obj._pimpl); }
        static inline std::shared_ptr<const impl::CuService_impl> get_ptr(const fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<const impl::CuService_impl>(obj._pimpl); }

        // std::atomic_flag destroy_sent;
        std::weak_ptr<CuService_impl> self;
        std::shared_ptr<fractos::core::channel> ch;
        fractos::core::cap::request req_make_device;
        const std::string name;
    };

    struct CuDevice_impl {
        // pimpl-to-impl
        static CuDevice_impl& get(fractos::service::compute::cuda::Device& obj)
            { return *reinterpret_cast<impl::CuDevice_impl*>(obj._pimpl.get()); }
        static const CuDevice_impl& get(const fractos::service::compute::cuda::Device& obj)
            { return *reinterpret_cast<impl::CuDevice_impl*>(obj._pimpl.get()); }

        std::weak_ptr<CuDevice_impl> self;
        std::shared_ptr<fractos::core::channel> ch;
        fractos::core::cap::request req_destroy;
    };

    



}
