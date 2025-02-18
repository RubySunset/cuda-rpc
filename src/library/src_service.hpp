// #pragma once

#include <fractos/service/compute/cuda.hpp>
#pragma once

#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <memory>

using namespace fractos;

namespace impl {

    class service {
        public:
            fractos::core::future<fractos::core::cap::request>
            register_methods(std::shared_ptr<fractos::core::channel> ch);
    
            void request_exit();
            bool exit_requested() const;
    
        protected:
            void handle_connect_cuda_service(auto ch, auto args);
            void handle_make_device(auto ch, auto args);
            friend class device;
    
        private:
            service(std::string name);
    
            std::weak_ptr<service> _self;
            std::atomic<bool> _requested_exit;
            std::string _name;
            fractos::core::cap::request _req_make_device;
    
            friend std::shared_ptr<service> make_cuda_service(std::string name);
            friend std::string to_string(const service& obj);
        };
    
        std::shared_ptr<service> make_cuda_service(std::string name);
    
        std::string to_string(const service& obj);
    
    // struct CuService_impl {
    //     CuService_impl(CuService_impl& other) = delete;
    //     CuService_impl(const CuService_impl& other) = delete;
        
    //     // pimpl-to-impl
    //     static inline impl::CuService_impl& get(fractos::service::compute::cuda::Service& obj)
    //         { return *reinterpret_cast<impl::CuService_impl*>(obj._pimpl.get()); }
    //     static inline const impl::CuService_impl& get(const fractos::service::compute::cuda::Service& obj)
    //         { return *reinterpret_cast<const impl::CuService_impl*>(obj._pimpl.get()); }

    //     static inline std::shared_ptr<impl::CuService_impl> get_ptr(fractos::service::compute::cuda::Service& obj)
    //         { return std::static_pointer_cast<impl::CuService_impl>(obj._pimpl); }
    //     static inline std::shared_ptr<const impl::CuService_impl> get_ptr(const fractos::service::compute::cuda::Service& obj)
    //         { return std::static_pointer_cast<const impl::CuService_impl>(obj._pimpl); }

    //     std::atomic_flag destroy_sent;
    //     public:
    //         fractos::core::future<fractos::core::cap::request>
    //         register_methods(std::shared_ptr<fractos::core::channel> ch);

    //         void request_exit();
    //         bool exit_requested() const;

    //     public:
    //         CuService_impl(std::string name);
    //         const std::string name;
    //         std::weak_ptr<CuService_impl> _self;
    //         fractos::core::cap::request _req_make_device;
    //     protected:
    //         void handle_connect_cuda_service(auto ch, auto args);
    //         // friend class object;
    
        
    //     private:
            
    //         friend std::shared_ptr<CuService_impl> make_service(std::string name);
    //         friend std::string to_string(const CuService_impl& obj);

    //         std::atomic<bool> _requested_exit;

    // };


    // std::shared_ptr<CuService_impl> make_service(std::string name);
    // std::string to_string(const CuService_impl& obj);


}
