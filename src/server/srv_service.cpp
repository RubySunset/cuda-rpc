// #include "device_service.hpp"
#include <glog/logging.h>
#include <srv_service.hpp>
// #include <srv_device.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_device_service::gpu_device_service() {

    // checkCudaErrors(cuInit(0));
}

void gpu_device_service::request_exit() {
   _requested_exit.store(true); 
}

bool gpu_device_service::exit_requested() const
{
    return _requested_exit.load();
}

std::shared_ptr<gpu_device_service> gpu_device_service::factory() {
    auto res = std::shared_ptr<gpu_device_service>(new gpu_device_service());
    res->_self = res;
    return res;
}

gpu_device_service::~gpu_device_service() {


}
/*
 *  The handler for make_device request
 *  Registers all methods that a Device has
 */
core::future<void> gpu_device_service::register_service(std::shared_ptr<core::channel> ch)
{
    // namespace msg_base = ::service::compute::detail::device_service;
    namespace msg_base = ::service::compute::cuda::wire::Service;

    auto self = _self.lock();

    return ch->make_request_builder<msg_base::make_device::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            // Call local service method with the actual handler
            // implementation. We must std::move args since it is a
            // unique_ptr.
            DVLOG(fractos::logging::SERVICE) << "In register_service handler";
            self->handle_make_device(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self, ch](auto& fut) {
                  self->req_make_device = fut.get();
                  DVLOG(fractos::logging::SERVICE) << "SET req_make_device"; // virtual
                  return ch->make_request_builder<msg_base::get_Device::request>(
                      ch->get_default_endpoint(),
                      [self](auto ch, auto args) {
                          self->handle_get_Device(std::move(args));
                      })
                      .on_channel()
                      .make_request();
        })
        .unwrap()
        .then([self](auto& fut) {
                  self->req_get_Device = fut.get();
                  DVLOG(fractos::logging::SERVICE) << "SET req_get_Device";
              });
}


/*
 *  Actual handler of the make_device request
 *  Initialize a Device and assign caps to it
 *  Return this Device to the frontend service
 */
void gpu_device_service::handle_make_device(auto args) {
    using msg = ::service::compute::cuda::wire::Service::make_device;
    
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

    wire::endian::uint8_t value = args->imms.value;

    auto self = _self.lock();

    VLOG(fractos::logging::SERVICE) << "vdev value is: " << (uint64_t)value;

    auto vdev = std::shared_ptr<gpu_Device>(gpu_Device::factory(value));

    vdev->register_methods(ch)
        .then([this, ch, self, vdev, args=std::move(args), value](auto& fut) {
            fut.get();
            _vdev = vdev;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .set_cap(&msg::response::caps::make_context, vdev->_req_make_context) // test
                .set_cap(&msg::response::caps::destroy, vdev->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();

}



void gpu_device_service::handle_get_Device(auto args) {
    // namespace msg_base = ::service::compute::cuda::wire::Service;
    using msg = ::service::compute::cuda::wire::Service::get_Device;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    auto vdev = _vdev;//_map[args->imms.value];

    if (args->has_exactly_args() and
        ch->has_object(vdev->_req_destroy).get()) {

        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
            .set_cap(&msg::response::caps::destroy, vdev->_req_destroy)
            .on_channel()
            .invoke()
            .as_callback();

    } else {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();
    }
}

std::string
test::to_string(const gpu_device_service& obj)
{
    std::stringstream ss;
    ss << "Service(" << &obj << ")";
    return ss.str();
}
