// #pragma once
#include <fractos/service/compute/cuda.hpp>


#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <./src_device.hpp>
#include <./common.hpp>


using namespace fractos;
using namespace impl;
// using namespace fractos::service::compute::cuda;

core::future<void>
impl::device::register_methods(std::shared_ptr<core::channel> ch)
{
    LOG_REQ("device register_methods");
    namespace msg = fractos::service::compute::cuda::message::device;

    auto self = this->self.lock();

    auto reqb_destroy = ch->make_request_builder<msg::destroy::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_destroy(ch, std::move(args));
        });

    return reqb_destroy.on_channel().make_request()
    .then([self](auto& fut) {
        self->req_destroy = fut.get();
    });
}

void
impl::device::handle_destroy(auto ch, auto args)
{
    /*
     * Revoke all capabilities, which effectively invalidates the object for all
     * remote clients; they cannot invoke any requests for the methods of this
     * object instance, while other objects still work as usual (each object has
     * independent method handler requests).
     *
     * After a method request is revoked, the handler function is destroyed. At
     * this point, the shared object pointer each handler holds is released, and
     * the object automatically deleted after the last of the handlers is gone.
     *
     * This approach ties method handling capabilities to the server-side
     * object, using the RAII pattern.
     */

    using msg = fractos::service::compute::cuda::message::device::destroy;

    if (not args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag)) {
        LOG_REQ("handle_destroy")
            << " [error] request without continuation, ignoring";
        return;
    }

    if (not args->has_exactly_args()) {

        LOG_REQ("handle_destroy")
            << " cont=" << args->caps.cont.get_cid()
            << " [error] malformed request";
        LOG_RES("handle_destroy")
            << " error=ERR_OTHER";

        ch->template make_request_builder<msg::response>(args->caps.cont)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("error invoking continuation, ignoring");

        return;
    }

    LOG_REQ("handle_destroy")
        << " cont=" << args->caps.cont.get_cid();

    // revoke each of the methods in turn, then invoke the response continuation

    auto self = this->self.lock();

    ch->revoke(self->req_destroy)
        .then([ch, self, args=std::move(args)](auto& fut) {
                  auto error = wire::ERR_SUCCESS;
                  try {
                      fut.get();
                  } catch (...) {
                      error = wire::exception_to_error(std::current_exception());
                  }

                  LOG_RES_PTR("handle_destroy", self)
                      << " error=" << wire::to_string(error);

                  ch->template make_request_builder<msg::response>(args->caps.cont)
                      .set_imm(&msg::response::imms::error, error)
                      .on_channel()
                      .invoke()
                      .as_callback_log_ignore_error("error invoking continuation, ignoring");
              })
        .as_callback();
}


impl::device::device(std::shared_ptr<service> srv, uint64_t value)
    :value(value)
{
}

std::shared_ptr<impl::device>
impl::make_device(std::shared_ptr<service> srv, uint64_t value)
{
    auto res = std::shared_ptr<device>(new device(srv, value));
    res->self = res;
    return res;
}

std::string
impl::to_string(const impl::device& obj)
{
    std::stringstream ss;
    ss << "device(" << &obj << ")";
    return ss.str();
}

