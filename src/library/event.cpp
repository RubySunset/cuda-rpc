#include <event_impl.hpp>
#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Event;
using namespace fractos;


#define IMPL_CLASS impl::Event
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Event>;


std::shared_ptr<clt::Event>
impl::make_event(std::shared_ptr<fractos::core::channel> ch,
                 CUevent cuevent,
                 fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::EventState>();
    state->req_generic = std::move(req_generic);
    state->cuevent = cuevent;

    return impl::Event::make(ch, state);
}

core::future<void>
impl::EventState::do_destroy(std::shared_ptr<core::channel>& ch)
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
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

CUevent
clt::Event::get_event() const
{
    auto& pimpl = impl::Event::get(*this);
    return pimpl.state->cuevent;
}

core::future<void>
clt::Event::synchronize()
{
    METHOD(synchronize);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Event::get(*this);
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


core::future<void>
clt::Event::record(Stream& stream)
{
    CUstream custream = stream.get_stream();

    METHOD(record);
    LOG_REQ(method)
        << " custream=" << (void*)custream;

    auto& pimpl = impl::Event::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_RECORD)
        .set_imm(&msg::request::imms::custream, (uint64_t)custream)
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


std::string
clt::to_string(const Event& obj)
{
    auto& pimpl = impl::Event::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const Event& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const EventState& obj)
{
    std::stringstream ss;
    ss << "cuda::Event(" << (void*)obj.cuevent << ")";
    return ss.str();
}
