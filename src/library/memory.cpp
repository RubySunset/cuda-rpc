
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include <memory_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Memory_impl& Memory_impl::get(Memory& obj)
{
    return *reinterpret_cast<Memory_impl*>(obj._pimpl.get());
}

inline
const Memory_impl& Memory_impl::get(const Memory& obj) 
{
    return *reinterpret_cast<Memory_impl*>(obj._pimpl.get());
}




Memory::Memory(std::shared_ptr<void> pimpl, wire::endian::uint64_t size) : _pimpl(pimpl) {



    DLOG(INFO) << "initialize memory : " << size;
}

char* Memory::get_addr() {
    auto& pimpl = Memory_impl::get(*this);
    

    return pimpl.addr;
}

core::cap::memory& Memory::get_cap_mem() {
    auto& pimpl = Memory_impl::get(*this);

    return pimpl.memory;
}


Memory::~Memory() {
    DLOG(INFO) << "Memory: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}


core::future<void> Memory::destroy() {
    using msg = ::service::compute::cuda::message::Memory::destroy;

    DVLOG(logging::SERVICE) << "Memory::destroy <-";

    auto& pimpl = Memory_impl::get(*this);
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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

