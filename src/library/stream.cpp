
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include <stream_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Stream_impl& Stream_impl::get(Stream& obj)
{
    return *reinterpret_cast<Stream_impl*>(obj._pimpl.get());
}

inline
const Stream_impl& Stream_impl::get(const Stream& obj) 
{
    return *reinterpret_cast<Stream_impl*>(obj._pimpl.get());
}

Stream::Stream(std::shared_ptr<void> pimpl, wire::endian::uint32_t flags, fractos::wire::endian::uint32_t id) : _pimpl(pimpl) {
    DLOG(INFO) << "initialize steam flag : " << flags;
}


Stream::~Stream() {
    DLOG(INFO) << "Stream: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}


fractos::wire::endian::uint32_t Stream::get_stream_id() {
    auto& pimpl = Stream_impl::get(*this);

    return pimpl.id;
}


core::future<void> Stream::synchronize() {
    using msg = ::service::compute::cuda::wire::Stream::synchronize;

    DVLOG(logging::SERVICE) << "Stream::synchronize <-";

    auto& pimpl = Stream_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}



core::future<void> Stream::destroy() {
    using msg = ::service::compute::cuda::wire::Stream::destroy;

    DVLOG(logging::SERVICE) << "Stream::destroy <-";

    auto& pimpl = Stream_impl::get(*this);
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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

