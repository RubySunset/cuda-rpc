#include "srv_device.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
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
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a cuda_device's caps
 */
core::future<void> gpu_cuda_device::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_device;


    auto self = _self;

    return ch->make_request_builder<msg_base::make_cuda_context::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            DLOG(INFO) << "In device register_methods handler";
            self->handle_make_cuda_context(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_make_cuda_context = fut.get();
            DLOG(INFO) << "SET req_make_cuda_context";
            return ch->make_request_builder<msg_base::destroy::request>(
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    self->handle_destroy(std::move(args));
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self, this](auto& fut) {
            DLOG(INFO) << "SET req_destroy device";
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a cuda_device, revoke all of its caps
 */
void gpu_cuda_device::handle_make_cuda_context(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle make_cuda_context";
    using msg = ::service::compute::cuda::message::cuda_device::make_cuda_context;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        DLOG(ERROR) << "got request without continuation, ignoring";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    if (not args->has_exactly_args()) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();
        return;
    }

    unsigned int value = args->imms.flags; // uint32_t

    auto self = _self; // lock()

    LOG(INFO) << "vctx value is: " << (uint64_t)value;

    auto vctx = std::shared_ptr<gpu_cuda_context>(gpu_cuda_context::factory(value));

    vctx->register_methods(ch)
        .then([this, ch, self, vctx, args=std::move(args), value](auto& fut) {
            fut.get();
            _vctx = vctx;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::make_cuda_Memalloc, vctx->_req_cuda_Memalloc)
                .set_cap(&msg::response::caps::destroy, vctx->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}


/*
 *  Destroy a cuda_device, revoke all of its caps
 */
void gpu_cuda_device::handle_destroy(auto args) {
    LOG(INFO) << "CALL handle destroy";
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

    LOG(INFO) << "Revoke destroy";

    ch->revoke(self->_req_make_cuda_context)
        .then([ch, self](auto& fut) {
                  fut.get();
                  LOG(INFO) << "Revoke _req_register_function";
                  return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            LOG(INFO) << "Virtual device destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}
