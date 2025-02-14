#pragma once

#include <fractos/service/compute/cuda.hpp>

using namespace fractos;

namespace impl {

    struct Service_impl {
        Service_impl(Service_impl& other) = delete;
        Service_impl(const Service_impl& other) = delete;

        static inline impl::Service_impl& get(fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<impl::Service_impl*>(obj._pimpl.get()); }
        static inline const impl::Service_impl& get(const fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<const impl::Service_impl*>(obj._pimpl.get()); }

        static inline std::shared_ptr<impl::Service_impl> get_ptr(fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<impl::Service_impl*>(obj._pimpl); }
        static inline std::shared_ptr<const impl::Service_impl> get_ptr(const fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<const impl::Service_impl*>(obj._pimpl); }

        std::atomic_flag destroy_sent;
    };

    std::shared_ptr<Service> make_service(std::shared_ptr<core::channel> ch,
        core::gns::service& gns, const std::string& name,
        const std::chrono::microseconds& wait_time = std::chrono::seconds{0});


}
