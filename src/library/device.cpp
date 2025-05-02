#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <utility>

#include <./common.hpp>
#include <device_impl.hpp>
#include <context_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
srv::to_string(const srv::Device& obj)
{
    auto& pimpl = impl::Device::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Device& obj)
{
    std::stringstream ss;
    ss << "cuda::Device(" << &obj << ")";
    return ss.str();
}

impl::Device::Device(std::shared_ptr<fractos::core::channel> channel,
                     CUdevice device,
                     fractos::core::cap::request req_generic,
                     fractos::core::cap::request req_make_context,
                     fractos::core::cap::request req_destroy)
    :ch(channel)
    ,device(device)
    ,req_generic(std::move(req_generic))
    ,req_make_context(std::move(req_make_context))
    ,req_destroy(std::move(req_destroy))
{
}

srv::Device::Device(std::shared_ptr<void> pimpl)
    :_pimpl(pimpl)
{
}

srv::Device::~Device()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
}

CUdevice
srv::Device::get_device() const
{
    auto& pimpl = impl::Device::get(*this);
    return pimpl.device;
}

core::future<int>
srv::Device::get_attribute(CUdevice_attribute attrib) const
{
    METHOD(Device, get_attribute);
    LOG_REQ(method)
        << " attrib=" << attrib;

    auto& pimpl = impl::Device::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Device::OP_GET_ATTRIBUTE)
        .set_imm(&msg::request::imms::attrib, attrib)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.pi;
        });
}

core::future<std::string>
srv::Device::get_name() const
{
    METHOD(Device, get_name);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Device::OP_GET_NAME)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_CAPS_EXACT();
            CHECK_ARGS_IMMS_MIN();
            if (args->imms_size() != (args->imms_expected_size() + args->imms.len)) {
                throw core::other_error("invalid response format for " + method);
            }
            CHECK_ARGS_ERROR();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return std::string(args->imms.name, args->imms.len);
        });
}

core::future<CUuuid>
srv::Device::get_uuid() const
{
    METHOD(Device, get_uuid);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Device::OP_GET_UUID)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            CUuuid uuid = *(CUuuid*)args->imms.uuid;
            return uuid;
        });
}

core::future<size_t>
srv::Device::total_mem() const
{
    METHOD(Device, total_mem);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Device::OP_TOTAL_MEM)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.bytes;
        });
}

core::future<std::shared_ptr<srv::Context>>
srv::Device::make_context(unsigned int flags)
{
    using msg = ::service::compute::cuda::wire::Device::make_context;

    DVLOG(logging::SERVICE) << "Device::make_context <-";

    auto& pimpl = impl::Device::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_make_context)
        .set_imm(&msg::request::imms::flags, flags) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([this, self=pimpl.self.lock(), flags](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for Device::make_context");
                DVLOG(logging::SERVICE) << "Device::make_context ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Device::make_context ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            auto pimpl_ = std::make_shared<impl::Context>(
                ch,
                self,
                std::move(args->caps.generic),
                std::move(args->caps.make_memory),
                std::move(args->caps.make_memory_rpc_test),
                std::move(args->caps.make_stream),
                std::move(args->caps.make_event),
                std::move(args->caps.make_module_data),
                // std::move(args->caps.make_module_file),
                std::move(args->caps.synchronize),
                std::move(args->caps.destroy));
            auto pimpl = static_pointer_cast<void>(pimpl_);
            auto res = std::make_shared<Context>(pimpl, flags);
            pimpl_->self = res;
            return res;
        });
}

core::future<void>
srv::Device::destroy()
{
    auto& pimpl = impl::Device::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Device::do_destroy()
{
    using msg = ::service::compute::cuda::wire::Device::destroy;

    DVLOG(logging::SERVICE) << "virtual_device::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Device::destroy");
                DVLOG(logging::SERVICE) << "Device::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Device::destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
