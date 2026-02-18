#include <cuda.h>
#include <cublas_v2.h>
#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/endian.hpp>
#include <fractos/wire/error.hpp>
#include <functional>
#include <stream_impl.hpp>
#include <utility>

#include "./cublas_impl.hpp"
#include "./context_impl.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::CublasHandle;
using namespace fractos;


#define IMPL_CLASS impl::CublasHandle
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::CublasHandle>;


std::shared_ptr<clt::CublasHandle>
impl::make_cublas_handle(clt::Context& ctx,
                  std::shared_ptr<fractos::core::channel> ch,
                  cublasHandle_t handle,
                  fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::CublasHandleState>();
    state->ctx = impl::Context::get(ctx).state->self;
    state->req_generic = std::move(req_generic);
    state->handle = handle;

    return impl::CublasHandle::make(ch, state);
}

std::string
clt::to_string(const clt::CublasHandle& obj)
{
    auto& pimpl = impl::CublasHandle::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::CublasHandle& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::CublasHandleState& obj)
{
    std::stringstream ss;
    ss << "cuda::CublasHandle(" << (void*)obj.handle << ")";
    return ss.str();
}

core::future<void>
impl::CublasHandleState::do_destroy(std::shared_ptr<core::channel>& ch)
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
        .then_check_cublas_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

std::shared_ptr<clt::Context>
clt::CublasHandle::get_context() const
{
    auto& pimpl = impl::CublasHandle::get(*this);

    return pimpl.state->ctx.lock();
}

cublasHandle_t
clt::CublasHandle::get_handle() const
{
    auto& pimpl = impl::CublasHandle::get(*this);

    return pimpl.state->handle;
}

core::future<void>
clt::CublasHandle::autogen_func(const void** args_ptr, const std::vector<size_t>& args_size, uint32_t func_id, Stream& stream)
{
    METHOD(autogen_func);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::CublasHandle::get(*this);
    auto self = pimpl.state->self.lock();

    CUstream custream = stream.get_stream();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req = pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_AUTOGEN_FUNC)
        .set_imm(&msg::request::imms::func_id, func_id)
        .set_imm(&msg::request::imms::custream, (uint64_t)custream)
        .set_cap(&msg::request::caps::continuation, resp);

    size_t offset = offsetof(msg::request::imms, args);
    for (size_t i = 0; i < args_size.size(); i++) {
        auto size = args_size[i];
        req.set_imm(offset, args_ptr[i], size);
        offset += size;
    }

    return req
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then_check_cublas_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}
