#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <memory_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Memory;
using namespace fractos;


#define IMPL_CLASS impl::Memory
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Memory>;


std::shared_ptr<clt::Memory>
impl::make_memory(std::shared_ptr<fractos::core::channel> ch,
                  CUdeviceptr addr,
                  size_t size,
                  core::cap::request req_mem_destroy,
                  core::cap::memory memory)
{
    auto state = std::make_shared<impl::MemoryState>();
    state->req_mem_destroy = std::move(req_mem_destroy);
    state->addr = addr;
    state->size = size;
    state->memory = std::move(memory);

    return impl::Memory::make(ch, state);
}

core::future<void>
impl::MemoryState::do_destroy(std::shared_ptr<core::channel>& ch)
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


char*
clt::Memory::get_addr()
{
    auto& pimpl = impl::Memory::get(*this);

    return (char*)pimpl.state->addr;
}

core::cap::memory&
clt::Memory::get_cap_mem()
{
    auto& pimpl = impl::Memory::get(*this);

    return pimpl.state->memory;
}
