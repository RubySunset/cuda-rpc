#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <memory_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


inline
impl::Memory&
impl::Memory::get(srv::Memory& obj)
{
    return *reinterpret_cast<impl::Memory*>(obj._pimpl.get());
}

inline
const impl::Memory&
impl::Memory::get(const srv::Memory& obj)
{
    return *reinterpret_cast<impl::Memory*>(obj._pimpl.get());
}




srv::Memory::Memory(std::shared_ptr<void> pimpl, fractos::wire::endian::uint64_t size)
    :_pimpl(pimpl)
{
    DLOG(INFO) << "initialize memory : " << size;
}

char*
srv::Memory::get_addr()
{
    auto& pimpl = impl::Memory::get(*this);

    return pimpl.addr;
}

core::cap::memory&
srv::Memory::get_cap_mem()
{
    auto& pimpl = impl::Memory::get(*this);

    return pimpl.memory;
}


srv::Memory::~Memory() {
    DLOG(INFO) << "Memory: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}


core::future<void>
srv::Memory::destroy()
{
    using msg = ::service::compute::cuda::wire::Memory::destroy;

    DVLOG(logging::SERVICE) << "Memory::destroy <-";

    auto& pimpl = impl::Memory::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_mem_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Memory::destroy");
                DVLOG(logging::SERVICE) << "Memory::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Memory::destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
