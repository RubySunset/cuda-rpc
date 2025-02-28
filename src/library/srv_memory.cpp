#include "srv_memory.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_memory::gpu_cuda_memory(fractos::wire::endian::uint32_t size) {
    //fork();
    _size = size;
    _destroyed = false;
   
}

std::shared_ptr<gpu_cuda_memory> gpu_cuda_memory::factory(fractos::wire::endian::uint32_t size){
    auto res = std::shared_ptr<gpu_cuda_memory>(new gpu_cuda_memory(size));
    res->_self = res;
    return res;
}

gpu_cuda_memory::~gpu_cuda_memory() {
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a cuda_memory's caps
 */
core::future<void> gpu_cuda_memory::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_memory;

    auto self = _self;


    return ch->make_request_builder<msg_base::destroy::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_destroy(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self, this](auto& fut) {
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a cuda_memory, revoke all of its caps
 */
void gpu_cuda_memory::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::message::cuda_memory::destroy;

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    
    auto self = this->_self;

    if (not args->has_exactly_args() or _destroyed) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();

        return;
    }

    DVLOG(logging::SERVICE) << "Revoke destroy";
                  
    ch->revoke(self->_req_destroy)
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DLOG(INFO) << "Virtual context destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}
