// #pragma once

#include <fractos/service/compute/cuda.hpp>

using namespace fractos;

namespace impl {

    struct Service_impl {
        Service_impl(Service_impl& other) = delete;
        Service_impl(const Service_impl& other) = delete;
        
        // pimpl-to-impl
        static inline impl::Service_impl& get(fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<impl::Service_impl*>(obj._pimpl.get()); }
        static inline const impl::Service_impl& get(const fractos::service::compute::cuda::Service& obj)
            { return *reinterpret_cast<const impl::Service_impl*>(obj._pimpl.get()); }

        static inline std::shared_ptr<impl::Service_impl> get_ptr(fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<impl::Service_impl>(obj._pimpl); }
        static inline std::shared_ptr<const impl::Service_impl> get_ptr(const fractos::service::compute::cuda::Service& obj)
            { return std::static_pointer_cast<const impl::Service_impl>(obj._pimpl); }

        std::atomic_flag destroy_sent;
        public:
            fractos::core::future<fractos::core::cap::request>
            register_methods(std::shared_ptr<fractos::core::channel> ch);

            void request_exit();
            bool exit_requested() const;

        public:
            Service_impl(std::string name);
            std::weak_ptr<Service_impl> _self;
        
        private:
            std::string _name;
            friend std::shared_ptr<Service_impl> make_service(std::string name);
            std::atomic<bool> _requested_exit;

    };


    std::shared_ptr<Service_impl> make_service(std::string name);


}
