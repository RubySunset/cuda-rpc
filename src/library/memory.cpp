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
                  CUdeviceptr cudeviceptr,
                  core::cap::memory memory,
                  core::cap::request req_generic)
{
    auto state = std::make_shared<impl::MemoryState>();
    state->cudeviceptr = cudeviceptr;
    state->memory = std::move(memory);
    state->req_generic = std::move(req_generic);

    return impl::Memory::make(ch, state);
}

core::future<void>
impl::MemoryState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();
    CHECK(self);

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response_ptr(self)
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


CUdeviceptr
clt::Memory::get_deviceptr()
{
    auto& pimpl = impl::Memory::get(*this);
    return pimpl.state->cudeviceptr;
}

core::cap::memory&
clt::Memory::get_cap_mem()
{
    auto& pimpl = impl::Memory::get(*this);
    return pimpl.state->memory;
}

std::string
clt::to_string(const clt::Memory& obj)
{
    auto& pimpl = impl::Memory::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const Memory& obj)
{
    return to_string(*obj.state);
}

std::string
impl::to_string(const MemoryState& obj)
{
    std::stringstream ss;
    ss << "cuda::Memory(" << (void*)obj.cudeviceptr << ")";
    return ss.str();
}
