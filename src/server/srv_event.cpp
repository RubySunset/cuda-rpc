#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "common.hpp"
#include "srv_event.hpp"


using namespace fractos;
using namespace ::test;
// using namespace impl;



gpu_Event::gpu_Event(fractos::wire::endian::uint32_t flags, CUcontext& ctx) {
    //fork();
    _flags = flags;
    _destroyed = false;
    _ctx = ctx;

    checkCudaErrors(cuCtxSetCurrent(_ctx));

    CUevent event;
    checkCudaErrors(cuEventCreate(&event, flags));

    _event = event;
}

std::shared_ptr<gpu_Event> gpu_Event::factory(fractos::wire::endian::uint32_t flags, 
                                       CUcontext& ctx){
    auto res = std::shared_ptr<gpu_Event>(new gpu_Event(flags,  ctx));
    res->_self = res;
    return res;
}

gpu_Event::~gpu_Event() {
    // checkCudaErrors(cuCtxDestroy(context));
}

// const CUevent& gpu_Event::getCUEvent() const
// {
//     return _event;
// }

// void gpu_Event::event_synchronize() {
//     checkCudaErrors(cuEventSynchronize(_event));
// }


void gpu_Event::event_destroy()
{
    checkCudaErrors(cuCtxSetCurrent(_ctx));

    // Clean up
    checkCudaErrors(cuEventDestroy(_event));
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a Event's caps
 */
core::future<void> gpu_Event::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Event;

    auto self = _self;

    // return ch->make_request_builder<msg_base::synchronize::request>(
    //     ch->get_default_endpoint(), 
    //     [self](auto ch, auto args) {
    //         self->handle_synchronize(std::move(args));
    //     })
    //     .on_channel()
    //     .make_request()
    //     .then([ch, self](auto& fut) {
    //         self->_req_sync = fut.get();
    //         VLOG(fractos::logging::SERVICE) << "SET re_destory"; 
    //         return ch->make_request_builder<msg_base::destroy::request>( // file
    //             ch->get_default_endpoint(), 
    //             [self](auto ch, auto args) {
                    
    //                 self->handle_destroy(std::move(args)); // file
    //             })
    //             .on_channel()
    //             .make_request();
    //         })
    //     .unwrap()
    //     .then([ch, self, this](auto& fut) {
    //         self->_req_destroy = fut.get();
    //     });

    return ch->make_request_builder<msg_base::destroy::request>( // file
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            
            self->handle_destroy(std::move(args)); // file
        })
        .on_channel()
        .make_request()
        .then([ch, self, this](auto& fut) {
            self->_req_destroy = fut.get();
        });

}


// void gpu_Event::handle_synchronize(auto args) {
//     VLOG(fractos::logging::SERVICE) << "CALL handle synchronize";
//     using msg = ::service::compute::cuda::wire::Event::synchronize;

//     if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
//         LOG(ERROR) << "no continuation";
//         return;
//     }

//     std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

//     event_synchronize();

//     ch->make_request_builder<msg::response>(args->caps.continuation)
//         .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
//         .on_channel()
//         .invoke()
//         .as_callback();
// }



/*
 *  Destroy a Event, revoke all of its caps
 */
void gpu_Event::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Event::destroy;

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    
    auto self = this->_self;

    if (not args->has_exactly_args() or _destroyed) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();

        return;
    }

    event_destroy();

    DVLOG(logging::SERVICE) << "Revoke destroy";

    // ch->revoke(self->_req_sync)
    //     .then([ch, self](auto& fut) {
    //         fut.get();
    //         VLOG(fractos::logging::SERVICE) << "Revoke _req_sync";
    //         return ch->revoke(self->_req_destroy); // file
    //     })
    //     .unwrap()
    //     .then([this, ch, self, args=std::move(args)](auto& fut) {
    //         fut.get();
    //         DVLOG(fractos::logging::SERVICE) << "cuda event destroyed";
    //         this->_destroyed = true;
    //         ch->make_request_builder<msg::response>(args->caps.continuation) // response
    //             .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
    //             .on_channel()
    //             .invoke()
    //             .as_callback();
    //     })
    // .as_callback();

    ch->revoke(self->_req_destroy) // file
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DVLOG(fractos::logging::SERVICE) << "cuda event destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

