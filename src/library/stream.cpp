#include <utility>

#include <fractos/wire/endian.hpp>
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <stream_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;

inline
impl::Stream&
impl::Stream::get(srv::Stream& obj)
{
    return *reinterpret_cast<impl::Stream*>(obj._pimpl.get());
}

inline
const impl::Stream&
impl::Stream::get(const srv::Stream& obj)
{
    return *reinterpret_cast<impl::Stream*>(obj._pimpl.get());
}

srv::Stream::Stream(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags,
                    fractos::wire::endian::uint32_t id)
    :_pimpl(pimpl)
{
    DLOG(INFO) << "initialize steam flag : " << flags;
}


srv::Stream::~Stream()
{
    DLOG(INFO) << "Stream: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}


fractos::wire::endian::uint32_t
srv::Stream::get_stream_id()
{
    auto& pimpl = impl::Stream::get(*this);

    return pimpl.id;
}


core::future<void>
srv::Stream::synchronize()
{
    using msg = ::service::compute::cuda::wire::Stream::synchronize;

    DVLOG(logging::SERVICE) << "Stream::synchronize <-";

    auto& pimpl = impl::Stream::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_stream_sync)
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



core::future<void>
srv::Stream::destroy()
{
    using msg = ::service::compute::cuda::wire::Stream::destroy;

    DVLOG(logging::SERVICE) << "Stream::destroy <-";

    auto& pimpl = impl::Stream::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_stream_destroy)
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
