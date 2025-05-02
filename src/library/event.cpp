
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <event_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
namespace srv = fractos::service::compute::cuda;


impl::Event::Event(std::shared_ptr<fractos::core::channel> ch,
                   fractos::wire::endian::uint8_t error,
                   fractos::core::cap::request req_event_destroy)
    :ch(ch)
    ,error(error)
    ,req_event_destroy(std::move(req_event_destroy))
{
}

srv::Event::Event(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags)
    :_pimpl(pimpl)
{
}

srv::Event::~Event()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
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



core::future<void>
srv::Event::destroy()
{
    auto& pimpl = impl::Event::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Event::do_destroy()
{
    using msg = ::service::compute::cuda::wire::Event::destroy;

    DVLOG(logging::SERVICE) << "Event::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_event_destroy)
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
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}

