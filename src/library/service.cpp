#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>

#include <./common.hpp>
#include <service_impl.hpp>
#include <device_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
srv::to_string(const srv::Service& obj)
{
    auto& pimpl = impl::Service::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Service& obj)
{
    std::stringstream ss;
    ss << "cuda::Service(" << &obj << ","  << obj.state.get() << ")";
    return ss.str();
}


srv::Service::Service(std::shared_ptr<void> pimpl)
    : _pimpl(pimpl)
{
}

std::shared_ptr<impl::Service>
impl::make_service(std::shared_ptr<core::channel> ch,
                   core::cap::request req_connect,
                   core::cap::request req_generic)
{
    auto state = std::make_shared<impl::ServiceState>(
        std::move(req_connect), std::move(req_generic));
    return impl::make_service(ch, state);
}

std::shared_ptr<impl::Service>
impl::make_service(std::shared_ptr<core::channel> ch,
                   std::shared_ptr<impl::ServiceState> state)
{
    auto res = std::make_shared<impl::Service>();
    res->self = res;
    res->ch = ch;
    res->state = state;
    return res;
}

core::future<void>
srv::Service::destroy()
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Service::do_destroy()
{
    LOG(FATAL) << "not implemented";
    return core::make_ready_future();
}


std::shared_ptr<core::channel>
srv::Service::get_default_channel() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.ch;
}

void
srv::Service::set_default_channel(std::shared_ptr<core::channel> ch)
{
    auto& pimpl = impl::Service::get(*this);
    pimpl.ch = ch;
}

std::unique_ptr<srv::Service>
srv::Service::with_default_channel(std::shared_ptr<core::channel> ch)
{
    auto& pimpl = impl::Service::get(*this);
    auto new_pimpl = impl::make_service(ch, pimpl.state);
    return std::make_unique<srv::Service>(new_pimpl);
}


core::future<std::unique_ptr<srv::Service>>
srv::make_service(std::shared_ptr<core::channel> ch,
                  core::gns::service& gns, const std::string& name,
                  const std::chrono::microseconds& wait_time)
{
    static const std::string method = "service::compute::cuda::make_service[gns]";

    LOG_OP(method)
        << " name=" << name
        << " <- {}";

    return gns.get_wait_for<core::cap::request>(ch, name, wait_time)
        .then([ch, name](auto& fut) {
            auto req = fut.get();
            LOG_OP(method)
                << " -> " << core::to_string(req);
            return make_service(ch, req)
                .then([req=std::move(req)](auto& fut) {
                    return fut.get();
                });
        })
        .unwrap();
}

core::future<std::unique_ptr<srv::Service>>
srv::make_service(std::shared_ptr<core::channel> ch,
                  const core::cap::request& connect)
{
    static const std::string method = "service::compute::cuda::make_service[connect]";
    using msg = srv::wire::Service::connect;

    LOG_OP(method)
        << " connect=" << core::to_string(connect)
        << " <- {}";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(connect)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_OP(method)
                << " -> " << wire::to_string(*args);

            if (not args->has_exactly_args()) {
                throw core::other_error(method + ": invalid response format");
            }
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            auto pimpl = impl::make_service(
                ch,
                std::move(args->caps.connect),
                std::move(args->caps.generic));
            return std::make_unique<Service>(pimpl);
        });
}

const core::cap::request&
srv::Service::get_connect() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.state->req_connect;
}

core::future<int>
srv::Service::get_driver_version()
{
    METHOD(Service, get_driver_version);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_GET_DRIVER_VERSION)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.value.get();
        });
}

core::future<void>
srv::Service::init(unsigned int flags)
{
    METHOD(Service, init);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_INIT)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
        });
}


core::future<std::shared_ptr<srv::Device>>
srv::Service::device_get(int ordinal)
{
    METHOD(Service, device_get);
    LOG_REQ(method)
        << " ordinal=" << ordinal;

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_DEVICE_GET)
        .set_imm(&msg::request::imms::ordinal, ordinal)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock(), ordinal](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            if ((int)args->imms.device.get() == -1) {
                return std::shared_ptr<srv::Device>(nullptr);
            }

            // get Device object
            auto pimpl_ = std::make_shared<impl::Device>(
                ch, args->imms.device,
                std::move(args->caps.generic),
                std::move(args->caps.make_context),
                std::move(args->caps.destroy));
            auto pimpl = static_pointer_cast<void>(pimpl_);
            auto res = std::make_shared<Device>(pimpl);
            pimpl_->self = res;
            return res;
        });
}

core::future<int>
srv::Service::device_get_count()
{
    METHOD(Service, device_get_count);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_DEVICE_GET_COUNT)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.count.get();
        });
}


core::future<CUmoduleLoadingMode>
srv::Service::module_get_loading_mode()
{
    METHOD(Service, module_get_loading_mode);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_MODULE_GET_LOADING_MODE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return static_cast<CUmoduleLoadingMode>(args->imms.mode.get());
        });
}
