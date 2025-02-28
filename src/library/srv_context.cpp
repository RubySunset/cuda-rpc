#include "srv_context.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>
// #include <fractos/service/compute/cuda.hpp>

// #include <fractos/service/compute/cuda.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_context::gpu_cuda_context(fractos::wire::endian::uint32_t value) {
    //fork();
    _id = value;
    _destroyed = false;
   
}

std::shared_ptr<gpu_cuda_context> gpu_cuda_context::factory(fractos::wire::endian::uint32_t value){
    auto res = std::shared_ptr<gpu_cuda_context>(new gpu_cuda_context(value));
    res->_self = res;
    return res;
}

gpu_cuda_context::~gpu_cuda_context() {
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a cuda_context's caps
 */
core::future<void> gpu_cuda_context::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_context;

    auto self = _self;


    return ch->make_request_builder<msg_base::make_cuda_Memalloc::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            LOG(INFO) << "In register_service context handler";
            self->handle_cuda_Memalloc(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_cuda_Memalloc = fut.get();
            LOG(INFO) << "SET req_cuda_Memalloc"; // virtua
            return ch->make_request_builder<msg_base::destroy::request>(
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_destroy(std::move(args));
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self, this](auto& fut) {
            LOG(INFO) << "SET req_destroy context";
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a cuda_context, revoke all of its caps
 */
void gpu_cuda_context::handle_cuda_Memalloc(auto args) {
    LOG(INFO) << "CALL handle handle_cuda_Memalloc";
    using msg = ::service::compute::cuda::message::cuda_context::make_cuda_Memalloc;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        DLOG(ERROR) << "got request without continuation, ignoring";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    if (not args->has_exactly_args()) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();
        return;
    }

    std::size_t size = args->imms.size; // uint64_t

    char* base = allocate_memory(size);//, context);

    // auto mr_ = ch->make_memory_region(base, size, core::memory_region::translation_type::PIN);
    // std::shared_ptr<typename decltype(mr_)::element_type> mr(std::move(mr_)); // element_type??
    // ch->make_memory(base, size, *mr)
    // .then([ch, args=std::move(args), size, this, base, mr](auto& fut) {

    // std::shared_ptr<typename decltype(args_)::element_type> args_(std::move(args));

    auto self = _self; // lock()
    // LOG(INFO) << "cuda device addr: " << (void*)base;
    LOG(INFO) << "mem size is: " << size;


    auto dev_mem = std::shared_ptr<gpu_cuda_memory>(gpu_cuda_memory::factory(size));

    // dev_mem->_memory = fut.get();
    // dev_mem->base = (char*)base;
    // dev_mem->_mr = mr;


    dev_mem->register_methods(ch)
        .then([this, ch, self, dev_mem, args=std::move(args), size](auto& fut) {
            fut.get();
            _dev_mem = dev_mem;

            // LOG(INFO) << "BACKEND memory size is " << dev_mem->_memory.get_size();

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                // .set_imm(&msg::response::imms::address, dev_mem->_memory.get_addr())
                // .set_cap(&msg::response::caps::memory, dev_mem->_memory)
                .set_cap(&msg::response::caps::destroy, dev_mem->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
            })
        .as_callback();


}

/*
 *  Destroy a cuda_context, revoke all of its caps
 */
void gpu_cuda_context::handle_destroy(auto args) {
    LOG(INFO) << "CALL handle destroy";
    using msg = ::service::compute::cuda::message::cuda_context::destroy;

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

    LOG(INFO) << "Revoke destroy";

    ch->revoke(self->_req_cuda_Memalloc)
        .then([ch, self](auto& fut) {
                  fut.get();
                  LOG(INFO) << "Revoke _req_register_function";
                  return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            LOG(INFO) << "Virtual context destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

                  
    // ch->revoke(self->_req_destroy)
    //     .then([this, ch, self, args=std::move(args)](auto& fut) {
    //         fut.get();
    //         DLOG(INFO) << "Virtual context destroyed";
    //         this->_destroyed = true;
    //         ch->make_request_builder<msg::response>(args->caps.continuation) // response
    //             .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
    //             .on_channel()
    //             .invoke()
    //             .as_callback();
    //     })
    // .as_callback();

}


char* gpu_cuda_context::allocate_memory(size_t size){//, CUcontext& context) {

    // Initialize the CUDA driver
    checkCudaErrors(cuInit(0));

    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, 0, device));

    CUdeviceptr d_A;
    // size_t size = 1024;

    // Allocate memory on the device
    checkCudaErrors(cuMemAlloc(&d_A, size));

    // Clean up
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuCtxDestroy(context));

    char* addr = nullptr;
    // checkCudaErrors(cuCtxSetCurrent(context));

    // if (type == static_cast<uint64_t>(ALLOC_DEVICE)) {
    //     CUdeviceptr p;
    //     checkCudaErrors(cuMemAlloc(&p, size));
    //     addr = (char*)p;
    // } else if (type == static_cast<uint64_t>(ALLOC_MANAGED)){
    //     CUdeviceptr p;
    //     checkCudaErrors(cuMemAllocManaged(&p, size, CU_MEM_ATTACH_GLOBAL));
    //     addr = (char*)p;
    // }

    // CUdeviceptr p;
    // checkCudaErrors(cuMemAlloc(&p, size));
    addr = (char*)d_A;
    //checkCudaErrors(cuCtxPopCurrent(nullptr));
    return addr;
}
