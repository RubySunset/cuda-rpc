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
                 fractos::core::cap::request req_event_destroy)
{
    auto state = std::make_shared<impl::EventState>();
    state->req_event_destroy = std::move(req_event_destroy);
    state->cuevent = cuevent;

    return impl::Event::make(ch, state);
}

core::future<void>
impl::EventState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_event_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response_ptr(self)
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


// core::future<void> Event::synchronize() {
//     using msg = ::service::compute::cuda::wire::Event::synchronize;

//     DVLOG(logging::SERVICE) << "Event::synchronize <-";

//     auto& pimpl = impl::Event::get(*this);

//     auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
//     return pimpl.ch->make_request_builder<msg::request>(pimpl.req_stream_sync)
//         .set_cap(&msg::request::caps::continuation, resp)
//         .on_channel()
//         .invoke(resp) // wait for handle_sync
//         .unwrap()
//         .then([](auto& fut) {
//             auto [ch, args] = fut.get();

//             if (not args->has_exactly_args()) {
//                 DVLOG(logging::SERVICE) << "Event::synchronize ->"
//                                 << " error=OTHER args";
//             }

//             DVLOG(logging::SERVICE) << "Event::synchronize ->"
//                                     << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
//             wire::error_raise_exception_maybe(args->imms.error);
//         });
// }


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
