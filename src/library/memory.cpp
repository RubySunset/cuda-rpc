#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <memory_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


impl::Memory::Memory(std::shared_ptr<fractos::core::channel> ch,
                     char* addr, size_t size,
                     fractos::core::cap::request req_mem_destroy,
                     fractos::core::cap::memory memory)
    :ch(ch)
    ,req_mem_destroy(std::move(req_mem_destroy))
    ,addr(addr)
    ,size(size)
    ,memory(std::move(memory))
{
}

srv::Memory::Memory(std::shared_ptr<void> pimpl)
    :_pimpl(pimpl)
{
}

srv::Memory::~Memory()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
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


core::future<void>
srv::Memory::destroy()
{
    auto& pimpl = impl::Memory::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Memory::do_destroy()
{
    using msg = ::service::compute::cuda::wire::Memory::destroy;

    DVLOG(logging::SERVICE) << "Memory::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_mem_destroy)
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
