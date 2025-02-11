#pragma once

#include <fractos/service/compute/cuda.hpp>


namespace impl {

    struct Service {
        Service(Service& other) = delete;
        Service(const Service& other) = delete;

        static inline impl::Service& get(fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<impl::Service*>(obj._pimpl.get()); }
        static inline const impl::Service& get(const fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<const impl::Service*>(obj._pimpl.get()); }

        static inline std::shared_ptr<impl::Service> get_ptr(fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<impl::Service*>(obj._pimpl); }
        static inline std::shared_ptr<const impl::Service> get_ptr(const fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<const impl::Service*>(obj._pimpl); }

        std::atomic_flag destroy_sent;
    };

    std::shared_ptr<Service>
    make_service();

}
