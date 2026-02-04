#include <cuda.h>
#include <pthread.h>

#include <glog/logging.h>

#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

#include "./common.hpp"
#include "./service.hpp"
#include "./context.hpp"
#include "./device.hpp"
#include "./stream.hpp"
#include "./event.hpp"
#include "./cublas.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::CublasHandle;
using namespace fractos;

extern bool route_autogen_func(uint32_t func_id, const char *args_ptr, cublasHandle_t handle, cublasStatus_t *cublas_error);


std::pair<std::shared_ptr<impl::CublasHandle>, cublasStatus_t>
impl::make_cublas_handle(std::shared_ptr<Context> ctx)
{
    std::shared_ptr<CublasHandle> res;
    cublasStatus_t cublas_error = CUBLAS_STATUS_SUCCESS;

    if (cuCtxSetCurrent(ctx->cucontext) != CUDA_SUCCESS) {
        cublas_error = CUBLAS_STATUS_NOT_INITIALIZED;
        goto out;
    }

    cublasHandle_t cublas_handle;
    cublas_error = cublasCreate_v2(&cublas_handle);
    if (cublas_error != CUBLAS_STATUS_SUCCESS) {
        goto out;
    }

    res = std::make_shared<CublasHandle>(ctx, cublas_handle);
    res->self = res;

out:
    return std::make_pair(res, cublas_error);
}

impl::CublasHandle::CublasHandle(std::shared_ptr<Context> ctx, cublasHandle_t cublas_handle)
    :cublas_handle(cublas_handle)
    ,ctx_ptr(ctx)
{
}

impl::CublasHandle::~CublasHandle()
{
}

std::string
impl::to_string(const impl::CublasHandle& obj)
{
    std::stringstream ss;
    ss << "CublasHandle(" << (void*)obj.get_remote_handle() << ")";
    return ss.str();
}

core::future<void>
impl::CublasHandle::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::CublasHandle;

    auto self = this->self;

    return ch->make_request_builder<srv_wire_msg::generic::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self](auto& fut) {
            self->req_generic = fut.get();
        });

}


cublasHandle_t
impl::CublasHandle::get_remote_handle() const
{
    return (cublasHandle_t)this;
}


void
impl::CublasHandle::handle_generic(auto ch, auto args)
{
    METHOD(generic);
    CHECK_CAPS_CONT(msg::request::caps::continuation);

    auto opcode = srv_wire_msg::OP_INVALID;
    if (args->has_imm(&msg::request::imms::opcode)) {
        opcode = static_cast<srv_wire_msg::generic_opcode>(args->imms.opcode.get());
    }

    auto reinterpreted = []<class T>(auto args) {
        using ptr = core::receive_args<T>;
        return std::unique_ptr<ptr>(reinterpret_cast<ptr*>(args.release()));
    };

#define CASE_HANDLE(NAME, name)                                         \
    case srv_wire_msg::OP_ ## NAME:                                      \
        handle_ ## name(ch, reinterpreted.template operator()<srv_wire_msg:: name ::request>(std::move(args))); \
        break;

    switch (opcode) {
    CASE_HANDLE(AUTOGEN_FUNC, autogen_func);
    CASE_HANDLE(DESTROY, destroy);
    default:
        LOG_RES(method)
            << " [error] invalid opcode";
        ch->template make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_continuation_error();
        break;
    }

#undef HANDLE
}

void
impl::CublasHandle::handle_autogen_func(auto ch, auto args)
{
    METHOD(autogen_func);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    auto cublas_error = CUBLAS_STATUS_SUCCESS;

    auto func_id = (uint32_t)args->imms.func_id.get();
    auto custream_arg = (CUstream)args->imms.custream.get();
    const char* args_ptr = args->imms.args;

    cuerror = cuCtxSetCurrent(ctx_ptr->cucontext);
    if (cuerror != CUDA_SUCCESS) {
        // Transform CUDA driver error into CUBLAS error
        // (this 'consumes' the CUDA driver error, resetting it to success)
        cuerror = CUDA_SUCCESS;
        cublas_error = CUBLAS_STATUS_NOT_INITIALIZED;
        goto out;
    }

    if (custream_arg) {
        auto stream_ptr = ctx_ptr->get_stream(custream_arg);
        if (!stream_ptr) {
            cuerror = CUDA_ERROR_INVALID_HANDLE;
            goto out;
        }
        cublasSetStream(cublas_handle, stream_ptr->custream);
    }

    if (!route_autogen_func(func_id, args_ptr, cublas_handle, &cublas_error)) {
        LOG_RES(method) << " [error] invalid func id";
        error = wire::ERR_OTHER;
    }

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror)
        << " cublas error=" << cublas_error;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .set_imm(&msg::response::imms::cublas_error, cublas_error)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

core::future<std::pair<wire::error_type, cublasStatus_t>>
impl::CublasHandle::destroy_maybe(auto ch)
{
    auto self = this->self;

    if (not common::service::SrvBase::destroy_maybe()) {
        auto error = wire::ERR_OTHER;
        return core::make_ready_future(std::make_pair(error, (cublasStatus_t)0));
    }

    return ch->revoke(self->req_generic)
        .then([this, self](auto& fut) {
            fut.get();

            auto error = wire::ERR_SUCCESS;
            auto cublas_error = CUBLAS_STATUS_SUCCESS;

            if (cuCtxSetCurrent(ctx_ptr->cucontext) != CUDA_SUCCESS) {
                cublas_error = CUBLAS_STATUS_NOT_INITIALIZED;
                goto out_inner;
            }

            cublas_error = cublasDestroy_v2(cublas_handle);
            if (cublas_error != CUBLAS_STATUS_SUCCESS) {
                goto out_inner;
            }

            out_inner:

            ctx_ptr->erase_cublas_handle(self);
            this->ctx_ptr.reset();
            this->self.reset();

            return std::make_pair(error, cublas_error);
        });
}

void
impl::CublasHandle::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;

    return destroy_maybe(ch)
        .then([ch, this, self, args=std::move(args)](auto& fut) {
            CUresult cuerror = CUDA_SUCCESS;
            auto [error, cublas_error] = fut.get();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " cublas error=" << cublas_error;
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::cublas_error, cublas_error)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();
}
