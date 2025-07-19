#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./context.hpp"
#include "./stream.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Stream;
using namespace fractos;


std::pair<CUresult, std::shared_ptr<impl::Stream>>
impl::make_stream(std::shared_ptr<Context> ctx, unsigned int flags)
{
    std::shared_ptr<Stream> res;

    auto error = cuCtxSetCurrent(ctx->cucontext);
    if (error != CUDA_SUCCESS) {
        return std::make_pair(error, res);
    }

    CUstream stream;
    error = cuStreamCreate(&stream, flags);
    if (error != CUDA_SUCCESS) {
        return std::make_pair(error, res);
    }

    res = std::make_shared<Stream>(ctx, stream);
    res->self = res;
    return std::make_pair(error, res);
}

impl::Stream::Stream(std::shared_ptr<Context> ctx, CUstream stream)
    :custream(stream)
    ,ctx_ptr(ctx)
{
}

impl::Stream::~Stream()
{
}

std::string
impl::to_string(const impl::Stream& obj)
{
    std::stringstream ss;
    ss << "Stream(" << (void*)obj.get_remote_custream() << ")";
    return ss.str();
}

core::future<void>
impl::Stream::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Stream;

    auto self = this->self.lock();
    CHECK(self);

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


CUstream
impl::Stream::get_remote_custream() const
{
    return (CUstream)this;
}


void
impl::Stream::handle_generic(auto ch, auto args)
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
    CASE_HANDLE(SYNCHRONIZE, synchronize);
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
impl::Stream::handle_synchronize(auto ch, auto args)
{
    METHOD(synchronize);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_IMMS_EXACT(reqb_cont);
    CHECK_CAPS_EXACT(reqb_cont);

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    cuerror = cuStreamSynchronize(custream);

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror);

    ch->template make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback();
}

void
impl::Stream::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self.lock();
    DCHECK(self);
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    if (not destroy_maybe()) {
        error = wire::ERR_OTHER;
        LOG_RES(method)
            << " error=" << wire::to_string(error)
            << " cuerror=" << get_CUresult_name(cuerror);
        ch->template make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, error)
            .set_imm(&msg::response::imms::cuerror, cuerror)
            .on_channel()
            .invoke()
            .as_callback();
        return;
    }

    auto ctx_ptr = this->ctx_ptr.lock();
    CHECK(ctx_ptr);
    ctx_ptr->erase_stream(self);

    cuerror = cuStreamDestroy(custream);

    ch->revoke(self->req_generic)
        .then([this, self, ch, args=std::move(args), error, cuerror](auto& fut) {
            fut.get();
            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror);
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();
}
