
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <event_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Event_impl& Event_impl::get(Event& obj)
{
    return *reinterpret_cast<Event_impl*>(obj._pimpl.get());
}

inline
const Event_impl& Event_impl::get(const Event& obj) 
{
    return *reinterpret_cast<Event_impl*>(obj._pimpl.get());
}

Event::Event(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags) : _pimpl(pimpl) {
    DLOG(INFO) << "initialize event flag : " << flags;
}


Event::~Event() {
    DLOG(INFO) << "Event: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

// core::future<void> Event::synchronize() {
//     using msg = ::service::compute::cuda::wire::Event::synchronize;

//     DVLOG(logging::SERVICE) << "Event::synchronize <-";

//     auto& pimpl = Event_impl::get(*this);

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



core::future<void> Event::destroy() {
    using msg = ::service::compute::cuda::wire::Event::destroy;

    DVLOG(logging::SERVICE) << "Event::destroy <-";

    auto& pimpl = Event_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_event_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Event::destroy");
                DVLOG(logging::SERVICE) << "Event::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Event::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

