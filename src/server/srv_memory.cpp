#include "srv_memory.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_Memory::gpu_Memory(fractos::wire::endian::uint32_t size, CUcontext& ctx) {
    //fork();
    _size = size;
    _destroyed = false;
    _ctx = ctx;
   
}

std::shared_ptr<gpu_Memory> gpu_Memory::factory(fractos::wire::endian::uint32_t size, CUcontext& ctx){
    auto res = std::shared_ptr<gpu_Memory>(new gpu_Memory(size, ctx));
    res->_self = res;
    return res;
}

gpu_Memory::~gpu_Memory() {
    // checkCudaErrors(cuCtxDestroy(context));
}


void gpu_Memory::memory_free(char* base)
{
    checkCudaErrors(cuCtxSetCurrent(_ctx));
    CUdeviceptr d_B = reinterpret_cast<CUdeviceptr>(base);

    // Clean up
    checkCudaErrors(cuMemFree(d_B));
    // checkCudaErrors(cuCtxDestroy(context));
}
/*
 *  Make handlers for a Memory's caps
 */
core::future<void> gpu_Memory::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Memory;

    auto self = _self;


    return ch->make_request_builder<msg_base::destroy::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_destroy(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self, this](auto& fut) {
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a Memory, revoke all of its caps
 */
void gpu_Memory::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Memory::destroy;

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

    memory_free(base);

    DVLOG(logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->_memory)
        .then([ch, self](auto& fut) {
                  fut.get();
                  DVLOG(fractos::logging::SERVICE) << "Revoke _req_deallocate";
                  return ch->revoke(self->_req_destroy);
              })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DVLOG(fractos::logging::SERVICE) << "cuda memory destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

