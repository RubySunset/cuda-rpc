#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include <./common.hpp>
#include <device_impl.hpp>
#include <context_impl.hpp>
#include <stream_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Device;
using namespace fractos;


#define IMPL_CLASS impl::Device
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Device>;


std::shared_ptr<clt::Device>
impl::make_device(std::shared_ptr<core::channel> ch,
                  CUdevice device,
                  core::cap::request req_generic)
{
    auto state = std::make_shared<impl::DeviceState>();
    state->device = device;
    state->req_generic = std::move(req_generic);

    return impl::Device::make(ch, state);
}

core::future<void>
impl::DeviceState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


CUdevice
clt::Device::get_device() const
{
    auto& pimpl = impl::Device::get(*this);
    return pimpl.state->device;
}

core::future<int>
clt::Device::get_attribute(CUdevice_attribute attrib) const
{
    METHOD(get_attribute);
    LOG_REQ(method)
        << " attrib=" << attrib;

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_ATTRIBUTE)
        .set_imm(&msg::request::imms::attrib, attrib)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return args->imms.pi;
        });
}

core::future<std::string>
clt::Device::get_name() const
{
    METHOD(get_name);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_NAME)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_CAPS_EXACT();
            CHECK_IMMS_ALL();
            if (args->imms_size() != (args->imms_expected_size() + args->imms.len)) {
                throw core::other_error("invalid response format for " + method);
            }

            return std::string(args->imms.name, args->imms.len);
        });
}

core::future<CUuuid>
clt::Device::get_uuid() const
{
    METHOD(get_uuid);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_UUID)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            CUuuid uuid = *(CUuuid*)args->imms.uuid;
            return uuid;
        });
}

core::future<size_t>
clt::Device::total_mem() const
{
    METHOD(total_mem);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_TOTAL_MEM)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return args->imms.bytes;
        });
}

core::future<cudaDeviceProp>
clt::Device::get_properties() const
{
    METHOD(get_properties);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_PROPERTIES)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_CAPS_EXACT();
            CHECK_IMMS_ALL();
            CHECK_ARGS_COND(args->imms.data_size == sizeof(cudaDeviceProp));
            CHECK_ARGS_COND(args->imms_size() == (sizeof(args->imms) + sizeof(cudaDeviceProp)));

            cudaDeviceProp res;
            memcpy(&res, args->imms.data, sizeof(res));
            return res;
        });
}

core::future<std::shared_ptr<clt::Context>>
clt::Device::make_context(const CUctxCreateParams& ctxCreateParams, unsigned int flags)
{
    METHOD(ctx_create);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Device::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_CTX_CREATE)
        .set_imm(&msg::request::imms::flags, flags)
        .set_imm(&msg::request::imms::ctxCreateParams, &ctxCreateParams, sizeof(CUctxCreateParams))
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([this, self, flags](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            auto context = impl::make_context(
                ch,
                self,
                (CUcontext)args->imms.cucontext.get(),
                std::move(args->caps.generic_ctx));

            auto legacy_default_stream = impl::make_stream(
                *context,
                ch,
                (CUstream)args->imms.legacy_default_custream.get(),
                std::move(args->caps.generic_legacy_default_stream));

            auto& pimpl = impl::Context::get(*context);
            pimpl.state->legacy_default_stream = legacy_default_stream;

            return context;
        });
}

core::future<std::shared_ptr<clt::Context>>
clt::Device::make_context(unsigned int flags)
{
    CUctxCreateParams ctxCreateParams = {};
    return make_context(ctxCreateParams, flags);
}

std::string
clt::to_string(const clt::Device& obj)
{
    auto& pimpl = impl::Device::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Device& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::DeviceState& obj)
{
    std::stringstream ss;
    ss << "cuda::Device(" << &obj << ")";
    return ss.str();
}
