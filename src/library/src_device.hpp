// #pragma once

#include <fractos/service/compute/cuda.hpp>
#pragma once

#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <memory>

using namespace fractos;

namespace impl {

    class service;

    
 
    
    class device{
        public:
            fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

            std::atomic<uint64_t> value;
            std::weak_ptr<device> self;
            fractos::core::cap::request req_make_context;
            fractos::core::cap::request req_destroy;

            fractos::core::cap::request req_fetch;
            fractos::core::cap::request req_add;
            fractos::core::cap::request req_sub;
            fractos::core::cap::request req_store;
            // fractos::core::cap::request req_destroy;



        protected: 
            // void handle_make_context(auto ch, auto args);
            void handle_fetch(auto ch, auto args);
            void handle_add(auto ch, auto args);
            void handle_sub(auto ch, auto args);
            void handle_store(auto ch, auto args);
    
            void handle_destroy(auto ch, auto args);
            // friend class context;
        
        private:
            device(std::shared_ptr<service> srv, uint64_t value);

            friend std::shared_ptr<device> make_device(std::shared_ptr<service> srv, uint64_t value);

    };

    //device
    std::shared_ptr<device> make_device(std::shared_ptr<service> srv, uint64_t value);
    std::string to_string(const device& obj);
    

}
