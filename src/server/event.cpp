#include <fractos/common/service/srv_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./service.hpp"
#include "./device.hpp"
#include "./context.hpp"
#include "./event.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Event;
using namespace fractos;


std::string
impl::to_string(const impl::Event& obj)
{
    std::stringstream ss;
    ss << "Event(" << (void*)obj.get_remote_cuevent() << ")";
    return ss.str();
}


fractos::core::future<std::tuple<wire::error_type, CUresult, std::shared_ptr<impl::Event>>>
impl::make_event(std::shared_ptr<fractos::core::channel> ch,
                 std::shared_ptr<Context> ctx, unsigned int flags)
{
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    std::shared_ptr<Event> res;

    cuerror = cuCtxSetCurrent(ctx->cucontext);
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    CUevent cuevent;
    cuerror = cuEventCreate(&cuevent, flags);
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    res = std::make_shared<Event>();
    res->cuevent = cuevent;
    res->ctx_ptr = ctx;

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
                LOG(FATAL) << "TODO: undo Event and return error";
            } else {
                res->self = res;
            }

            res->ctx_ptr->insert_event(res);
            res->ctx_ptr->device->service->insert_event(res);

            return std::make_tuple(error, cuerror, res);
        });
}


CUevent
impl::Event::get_remote_cuevent() const
{
    return (CUevent)this;
}


void
impl::Event::handle_generic(auto ch, auto args)
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


core::future<std::tuple<wire::error_type, CUresult>>
impl::Event::destroy_maybe(auto ch)
{
    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    if (not common::service::SrvBase::destroy_maybe()) {
        error = wire::ERR_OTHER;
        return core::make_ready_future(std::make_tuple(error, cuerror));
    }

    return ch->revoke(req_generic)
        .then([ch, this, self](auto& fut) {
            fut.get();

            auto error = wire::ERR_SUCCESS;
            auto cuerror = cuCtxSetCurrent(ctx_ptr->cucontext);
            if (cuerror != CUDA_SUCCESS) {
                goto out_inner;
            }

            cuerror = cuEventDestroy(cuevent);
            if (cuerror != CUDA_SUCCESS) {
                goto out_inner;
            }

            out_inner:

            ctx_ptr->erase_event(self);
            ctx_ptr->device->service->erase_event(self);
            this->ctx_ptr.reset();
            this->self.reset();

            return std::make_tuple(error, cuerror);
        });
}

void
impl::Event::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;

    return destroy_maybe(ch)
        .then([ch, this, self, args=std::move(args)](auto& fut) {
            auto [error, cuerror] = fut.get();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror);
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();
}
