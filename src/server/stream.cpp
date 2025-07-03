#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "common.hpp"
#include "./context.hpp"
#include "./stream.hpp"


using namespace fractos;



std::pair<CUresult, std::shared_ptr<impl::Stream>>
impl::make_stream(Context& ctx, unsigned int flags)
{
    std::shared_ptr<Stream> res;

    auto error = cuCtxSetCurrent(ctx._ctx);
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

impl::Stream::Stream(Context& ctx, CUstream stream)
    :stream(stream)
    ,ctx_ptr(ctx._self)
{
}

impl::Stream::~Stream()
{
}

/*
 *  Make handlers for a Stream's caps
 */
core::future<void> impl::Stream::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Stream;

    auto self = this->self.lock();
    CHECK(self);

    return ch->make_request_builder<msg_base::synchronize::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_synchronize(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_sync = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET re_destory"; 
            return ch->make_request_builder<msg_base::destroy::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_destroy(std::move(args)); // file
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self, this](auto& fut) {
            self->_req_destroy = fut.get();
        });

}


void impl::Stream::handle_synchronize(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle synchronize";
    using msg = ::service::compute::cuda::wire::Stream::synchronize;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    auto cuerror = cuStreamSynchronize(stream);
    CHECK(cuerror == CUDA_SUCCESS);

    ch->make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
}



/*
 *  Destroy a Stream, revoke all of its caps
 */
void impl::Stream::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Stream::destroy;

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    
    auto self = this->self.lock();
    CHECK(self);

    if (not args->has_exactly_args() or not destroy_maybe()) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();

        return;
    }

    auto cuerr = cuStreamDestroy(stream);
    CHECK(cuerr == CUDA_SUCCESS);

    DVLOG(logging::SERVICE) << "Revoke destroy";

    // ch->revoke(self->_memory)
    //     .then([ch, self](auto& fut) {
    //               fut.get();
    //               DVLOG(fractos::logging::SERVICE) << "Revoke _req_deallocate";
    //               return ch->revoke(self->_req_destroy);
    //           })
    //     .unwrap()

    ch->revoke(self->_req_sync)
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_sync";
            return ch->revoke(self->_req_destroy); // file
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DVLOG(fractos::logging::SERVICE) << "cuda stream destroyed";
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

