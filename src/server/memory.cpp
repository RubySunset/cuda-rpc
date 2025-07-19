#include <fractos/common/service/srv_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./context.hpp"
#include "./memory.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Memory;
using namespace fractos;


fractos::core::future<std::tuple<wire::error_type, CUresult, std::shared_ptr<impl::Memory>>>
impl::make_memory(std::shared_ptr<core::channel> ch, std::shared_ptr<Context> ctx, size_t size)
{
    using res_type = decltype(make_memory(ch, ctx, size));

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    std::shared_ptr<Memory> res;

    cuerror = cuCtxSetCurrent(ctx->cucontext);
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    CUdeviceptr cuptr = 0;
    cuerror = cuMemAlloc(&cuptr, size);
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    res = std::make_shared<Memory>();
    res->cuptr = cuptr;
    // NOTE: default mr leads to failed RDMA access error
    res->mr = ch->make_memory_region((void*)cuptr, size, fractos::core::memory_region::translation_type::PIN);
    res->ctx = ctx;

    return ch->make_memory((void*)cuptr, size, *res->mr)
        .then([ch, error, cuerror, res](auto& fut) mutable -> res_type {
            try {
                res->memory = fut.get();
            } catch (const core::generic_error& e) {
                error = (wire::error_type)e.error;
                return core::make_ready_future(std::make_tuple(error, cuerror, res));
            }

            return ch->make_request_builder<srv_wire_msg::generic::request>(
                ch->get_default_endpoint(),
                [res](auto ch, auto args) {
                    res->handle_generic(ch, std::move(args));
                })
                .on_channel()
                .make_request()
                .then([res](auto& fut) {
                    res->req_generic = fut.get();
                })
                .then([error, cuerror, res](auto& fut) mutable {
                    try {
                        fut.get();
                    } catch (const core::generic_error& e) {
                        error = (wire::error_type)e.error;
                    }

                    if (error or cuerror) {
                        LOG(FATAL) << "TODO: undo Memory and return error";
                    } else {
                        res->self = res;
                    }

                    return std::make_tuple(error, cuerror, res);
                });
        })
        .unwrap();
}


std::string
impl::to_string(const impl::Memory& obj)
{
    std::stringstream ss;
    ss << "Memory(" << (void*)obj.cuptr << ")";
    return ss.str();
}


void
impl::Memory::handle_generic(auto ch, auto args)
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

#undef CASE_HANDLE
}

void
impl::Memory::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto ctx = this->ctx.lock();
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    if (not destroy_maybe()) {
        error = wire::ERR_OTHER;
        goto out;
    }

    ch->revoke(req_generic)
        .then([ch, this, self, args=std::move(args), ctx, error, cuerror](auto& fut) mutable {
            if (not ctx) {
                cuerror = CUDA_ERROR_INVALID_CONTEXT;
                goto out_inner;
            }

            cuerror = cuCtxSetCurrent(ctx->cucontext);
            if (cuerror != CUDA_SUCCESS) {
                goto out_inner;
            }

            cuerror = cuMemFree(cuptr);
            if (cuerror != CUDA_SUCCESS) {
                goto out_inner;
            }

            out_inner:
            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror);
            fut.get();
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();
    return;

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror);
    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}
