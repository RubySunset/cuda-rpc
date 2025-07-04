#include <utility>

#include <fractos/wire/endian.hpp>
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <stream_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Stream;
using namespace fractos;


#define IMPL_CLASS impl::Stream
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Stream>;


std::shared_ptr<clt::Stream>
impl::make_stream(std::shared_ptr<fractos::core::channel> ch,
                  fractos::wire::endian::uint32_t id,
                  fractos::core::cap::request req_generic,
                  fractos::core::cap::request req_stream_sync,
                  fractos::core::cap::request req_stream_destroy)
{
    auto state = std::make_shared<impl::StreamState>();
    state->req_generic = std::move(req_generic);
    state->req_stream_sync = std::move(req_stream_sync);
    state->req_stream_destroy = std::move(req_stream_destroy);
    state->id = id;

    return impl::Stream::make(ch, state);
}

core::future<void>
impl::StreamState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    using msg = ::service::compute::cuda::wire::Stream::destroy;

    DVLOG(logging::SERVICE) << "Stream::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_stream_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Stream::destroy");
                DVLOG(logging::SERVICE) << "Stream::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Stream::destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}


fractos::wire::endian::uint32_t
clt::Stream::get_stream_id()
{
    auto& pimpl = impl::Stream::get(*this);

    return pimpl.state->id;
}


core::future<void>
clt::Stream::synchronize()
{
    using msg = ::service::compute::cuda::wire::Stream::synchronize;

    DVLOG(logging::SERVICE) << "Stream::synchronize <-";

    auto& pimpl = impl::Stream::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_stream_sync)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_sync
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                DVLOG(logging::SERVICE) << "Stream::synchronize ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Stream::synchronize ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
