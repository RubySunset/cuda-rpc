#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include "./kernel_impl.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Kernel;
using namespace fractos;


#define IMPL_CLASS impl::Kernel
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Kernel>;


std::shared_ptr<clt::Kernel>
impl::make_kernel(std::shared_ptr<fractos::core::channel> ch,
                  CUkernel cukernel,
                  fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::KernelState>();
    state->req_generic = std::move(req_generic);
    state->cukernel = cukernel;

    return impl::Kernel::make(ch, state);
}

core::future<void>
impl::KernelState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

CUkernel
clt::Kernel::get_kernel() const
{
    auto& pimpl = impl::Kernel::get(*this);
    return pimpl.state->cukernel;
}


std::string
clt::to_string(const Kernel& obj)
{
    auto& pimpl = impl::Kernel::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const Kernel& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const KernelState& obj)
{
    std::stringstream ss;
    ss << "cuda::Kernel(" << (void*)obj.cukernel << ")";
    return ss.str();
}
