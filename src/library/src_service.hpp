// #pragma once

#include <fractos/service/compute/cuda.hpp>
#pragma once

#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <memory>

using namespace fractos;

namespace impl {
    class service;
    class device;

    //service
    std::shared_ptr<service> make_cuda_service(std::string name);
    std::string to_string(const service& obj);
    //device
    std::shared_ptr<device> make_device(std::shared_ptr<service> srv, uint64_t value);
    std::string to_string(const device& obj);


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
            fractos::core::cap::request _req_make_cuda_device;
    
            friend std::shared_ptr<service> make_cuda_service(std::string name);
            friend std::string to_string(const service& obj);
    };
    
    
    
    class device{
        public:
            fractos::core::future<void> register_device_methods(std::shared_ptr<fractos::core::channel> ch);

            std::atomic<uint64_t> value;
            std::weak_ptr<device> self;
            fractos::core::cap::request req_make_context;
            fractos::core::cap::request req_destroy;

        protected: 
            void handle_make_context(auto ch, auto args);
            void handle_destroy(auto ch, auto args);
            // friend class context;
        
        private:
            device(std::shared_ptr<service> srv, uint64_t value);

            friend std::shared_ptr<device> make_device(std::shared_ptr<service> srv, uint64_t value);

    };
    

}
