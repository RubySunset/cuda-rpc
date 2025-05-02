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


impl::Stream::Stream(std::shared_ptr<fractos::core::channel> ch,
                     fractos::wire::endian::uint8_t error,
                     fractos::wire::endian::uint32_t id,
                     fractos::core::cap::request req_stream_sync,
                     fractos::core::cap::request req_stream_destroy)
    :ch(ch)
    ,error(error)
    ,req_stream_sync(std::move(req_stream_sync))
    ,req_stream_destroy(std::move(req_stream_destroy))
    ,id(id)
{
}

srv::Stream::Stream(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags,
                    fractos::wire::endian::uint32_t id)
    :_pimpl(pimpl)
{
}


srv::Stream::~Stream()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
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
    auto& pimpl = impl::Stream::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Stream::do_destroy()
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
