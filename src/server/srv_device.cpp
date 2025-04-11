#include "srv_device.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_Device::gpu_Device(wire::endian::uint8_t value) {
    //fork();
    _id = value;
    _destroyed = false;

    checkCudaErrors(cuInit(0));

    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, value));
    
    _device = device;
   
}

std::shared_ptr<gpu_Device> gpu_Device::factory(wire::endian::uint8_t value){
    auto res = std::shared_ptr<gpu_Device>(new gpu_Device(value));
    res->_self = res;
    return res;
}

gpu_Device::~gpu_Device() {

}

/*
 *  Make handlers for a Device's caps
 */
core::future<void> gpu_Device::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Device;


    auto self = _self;

    return ch->make_request_builder<msg_base::make_context::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            DVLOG(fractos::logging::SERVICE) << "In device register_methods handler";
            self->handle_make_context(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_make_context = fut.get();
            DVLOG(fractos::logging::SERVICE) << "SET req_make_context";
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
            DVLOG(fractos::logging::SERVICE) << "SET req_destroy device";
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a Device, revoke all of its caps
 */
void gpu_Device::handle_make_context(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle make_context";
    using msg = ::service::compute::cuda::wire::Device::make_context;
    
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

    VLOG(fractos::logging::SERVICE) << "vctx value is: " << (uint64_t)value;

    auto vctx = std::shared_ptr<gpu_Context>(gpu_Context::factory(value, _device));

    vctx->register_methods(ch)
        .then([this, ch, self, vctx, args=std::move(args), value](auto& fut) {
            fut.get();
            _vctx = vctx;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::make_memory, vctx->_req_memory)
                .set_cap(&msg::response::caps::make_memory_rpc_test, vctx->_req_memory_rpc_test)
                .set_cap(&msg::response::caps::make_stream, vctx->_req_stream)
                .set_cap(&msg::response::caps::make_event, vctx->_req_event)
                .set_cap(&msg::response::caps::make_module_data, vctx->_req_module_data) // data
                .set_cap(&msg::response::caps::synchronize, vctx->_req_synchronize)
                .set_cap(&msg::response::caps::destroy, vctx->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}


/*
 *  Destroy a Device, revoke all of its caps
 */
void gpu_Device::handle_destroy(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Device::destroy;

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

    VLOG(fractos::logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->_req_make_context)
        .then([ch, self](auto& fut) {
                  fut.get();
                  VLOG(fractos::logging::SERVICE) << "Revoke _req_register_function";
                  return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Virtual device destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}
