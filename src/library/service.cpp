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
    ss << "cuda::Service(" << &obj << ")";
    return ss.str();
}

inline
impl::Service&
impl::Service::get(srv::Service& obj)
{
    return *reinterpret_cast<impl::Service*>(obj._pimpl.get());
}

inline
const impl::Service&
impl::Service::get(const srv::Service& obj)
{
    return *reinterpret_cast<impl::Service*>(obj._pimpl.get());
}


impl::Service::Service(std::shared_ptr<fractos::core::channel> ch,
                       fractos::core::cap::request req_connect,
                       fractos::core::cap::request req_generic)
    :ch(ch)
    ,req_connect(std::move(req_connect))
    ,req_generic(std::move(req_generic))
{
}

srv::Service::Service(std::shared_ptr<void> pimpl)
    : _pimpl(pimpl)
{
}


std::shared_ptr<core::channel>
srv::Service::get_default_channel()
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.ch;
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

            auto pimpl_ = std::make_shared<impl::Service>(
                ch,
                std::move(args->caps.connect),
                std::move(args->caps.generic));
            pimpl_->self = pimpl_;
            auto pimpl = std::static_pointer_cast<void>(pimpl_);
            std::unique_ptr<Service> res(new Service{pimpl});

            return res;
        });
}

const core::cap::request&
srv::Service::get_connect() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.req_connect;
}

#define METHOD(name)                                                    \
    static const std::string method = #name;                            \
    using msg = ::service::compute::cuda::wire::Service:: name;

#define CHECK_RESP()                                                    \
    if (not args->has_exactly_args()) {                                 \
        throw core::other_error("invalid response format for " + method); \
    }                                                                   \
    fractos::wire::error_raise_exception_maybe(args->imms.error);

core::future<int>
srv::Service::get_driver_version()
{
    METHOD(get_driver_version);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_GET_DRIVER_VERSION)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_RESP();

            return args->imms.value.get();
        });
}

core::future<void>
srv::Service::init(unsigned int flags)
{
    METHOD(init);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
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
            CHECK_RESP();
        });
}


core::future<std::shared_ptr<srv::Device>>
srv::Service::device_get(int ordinal)
{
    METHOD(device_get);
    LOG_REQ(method)
        << " ordinal=" << ordinal;

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
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
            CHECK_RESP();

            // get Device object
            auto pimpl_ = std::make_shared<impl::Device>(
                ch, args->imms.device,
                std::move(args->caps.make_context),
                std::move(args->caps.destroy));
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            return std::make_shared<Device>(pimpl);
        });
}

core::future<int>
srv::Service::device_get_count()
{
    METHOD(device_get_count);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_DEVICE_GET_COUNT)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_RESP();

            return args->imms.count.get();
        });
}


core::future<CUmoduleLoadingMode>
srv::Service::module_get_loading_mode()
{
    METHOD(module_get_loading_mode);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Service::OP_MODULE_GET_LOADING_MODE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_RESP();

            return static_cast<CUmoduleLoadingMode>(args->imms.mode.get());
        });
}
