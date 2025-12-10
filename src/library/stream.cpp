#include <cuda.h>
#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/endian.hpp>
#include <fractos/wire/error.hpp>
#include <stream_impl.hpp>
#include <utility>

#include "./context_impl.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Stream;
using namespace fractos;


#define IMPL_CLASS impl::Stream
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Stream>;


std::shared_ptr<clt::Stream>
impl::make_stream(clt::Context& ctx,
                  std::shared_ptr<fractos::core::channel> ch,
                  CUstream custream,
                  fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::StreamState>();
    state->ctx = impl::Context::get(ctx).state->self;
    state->req_generic = std::move(req_generic);
    state->custream = custream;

    return impl::Stream::make(ch, state);
}

std::string
clt::to_string(const clt::Stream& obj)
{
    auto& pimpl = impl::Stream::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Stream& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::StreamState& obj)
{
    std::stringstream ss;
    ss << "cuda::Stream(" << (void*)obj.custream << ")";
    return ss.str();
}

core::future<void>
impl::StreamState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


std::shared_ptr<clt::Context>
clt::Stream::get_context() const
{
    auto& pimpl = impl::Stream::get(*this);

    return pimpl.state->ctx.lock();
}

CUstream
clt::Stream::get_stream() const
{
    auto& pimpl = impl::Stream::get(*this);

    return pimpl.state->custream;
}


core::future<void>
clt::Stream::wait_event(Event& event, CUevent_wait_flags flags)
{
    METHOD(wait_event);
    LOG_REQ(method)
        << " event=" << (void*)event.get_event()
        << " flags=" << flags;

    auto& pimpl = impl::Stream::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_WAIT_EVENT)
        .set_imm(&msg::request::imms::cuevent, (uint64_t)event.get_event())
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
clt::Stream::wait_value_32(CUdeviceptr addr, uint32_t value, unsigned int flags)
{
    METHOD(wait_value_32);
    LOG_REQ(method)
        << " addr=" << (void*)addr
        << " value=" << value
        << " flags=" << flags;

    auto& pimpl = impl::Stream::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_WAIT_VALUE_32)
        .set_imm(&msg::request::imms::addr, addr)
        .set_imm(&msg::request::imms::value, value)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
clt::Stream::write_value_32(CUdeviceptr addr, uint32_t value, unsigned int flags)
{
    METHOD(write_value_32);
    LOG_REQ(method)
        << " addr=" << (void*)addr
        << " value=" << value
        << " flags=" << flags;

    auto& pimpl = impl::Stream::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_WRITE_VALUE_32)
        .set_imm(&msg::request::imms::addr, addr)
        .set_imm(&msg::request::imms::value, value)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
clt::Stream::synchronize()
{
    METHOD(synchronize);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Stream::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_SYNCHRONIZE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}
