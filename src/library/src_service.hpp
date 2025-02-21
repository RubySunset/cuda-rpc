// #pragma once

#include <fractos/service/compute/cuda.hpp>
#pragma once

#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <memory>

using namespace fractos;

namespace impl {
    class device;


    //service
    



    class service {
        public:
            fractos::core::future<fractos::core::cap::request>
            register_methods(std::shared_ptr<fractos::core::channel> ch);
    
            void request_exit();
            bool exit_requested() const;
    
        protected:
            void handle_connect_service(auto ch, auto args);
            void handle_make_device(auto ch, auto args);
            friend class object;
    
        private:
            service(std::string name);
    
            std::weak_ptr<service> _self;
            std::atomic<bool> _requested_exit;
            std::string _name;
            fractos::core::cap::request _req_make_device;
    
            friend std::shared_ptr<service> make_service(std::string name);
            friend std::string to_string(const service& obj);
        };

    std::shared_ptr<service> make_service(std::string name);
    std::string to_string(const service& obj);
}
