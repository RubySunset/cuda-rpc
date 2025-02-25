#include "srv_device.hpp"
#include <pthread.h>

#include <fractos/wire/error.hpp>
using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_device::gpu_cuda_device(wire::endian::uint8_t value) {
    //fork();
    _id = value;
    _destroyed = false;
}


std::shared_ptr<gpu_cuda_device> gpu_cuda_device::factory(wire::endian::uint8_t value){
    auto res = std::shared_ptr<gpu_cuda_device>(new gpu_cuda_device(value));
    res->_self = res;
    return res;
}

gpu_cuda_device::~gpu_cuda_device() {
}

/*
 *  Make handlers for a cuda_device's caps
 */
core::future<void> gpu_cuda_device::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_device;


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
 *  Destroy a cuda_device, revoke all of its caps
 */
void gpu_cuda_device::handle_destroy(auto args) {
    using msg = ::service::compute::cuda::message::cuda_device::destroy;

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

    DLOG(INFO) << "Revoke destroy";

    ch->revoke(self->_req_destroy)
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DLOG(INFO) << "Virtual device destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}
