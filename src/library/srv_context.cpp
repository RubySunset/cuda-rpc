#include "srv_context.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_context::gpu_cuda_context(fractos::wire::endian::uint32_t value) {
    //fork();
    _id = value;
    _destroyed = false;
   
}

std::shared_ptr<gpu_cuda_context> gpu_cuda_context::factory(fractos::wire::endian::uint32_t value){
    auto res = std::shared_ptr<gpu_cuda_context>(new gpu_cuda_context(value));
    res->_self = res;
    return res;
}

gpu_cuda_context::~gpu_cuda_context() {
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a cuda_context's caps
 */
core::future<void> gpu_cuda_context::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_context;

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
 *  Destroy a cuda_context, revoke all of its caps
 */
void gpu_cuda_context::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::message::cuda_context::destroy;

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
