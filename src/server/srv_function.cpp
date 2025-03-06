#include "srv_function.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_function::gpu_cuda_function(std::string func_name, CUcontext& ctx, CUmodule& mod) {
    //fork();
    _name = func_name;
    _destroyed = false;
    _ctx = ctx;
    _mod = mod;
   
    CUfunction function;
    checkCudaErrors(cuModuleGetFunction(&function, mod, func_name.c_str()));
    _func = function;

}

std::shared_ptr<gpu_cuda_function> gpu_cuda_function::factory(std::string func_name, CUcontext& ctx, CUmodule& mod){
    auto res = std::shared_ptr<gpu_cuda_function>(new gpu_cuda_function(func_name, ctx, mod));
    res->_self = res;
    return res;
}

gpu_cuda_function::~gpu_cuda_function() {
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a cuda_function's caps
 */
core::future<void> gpu_cuda_function::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_function;

    auto self = _self;


    return ch->make_request_builder<msg_base::call::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_call(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_call = fut.get();
            LOG(INFO) << "SET req_call"; // virtua
            return ch->make_request_builder<msg_base::func_destroy::request>(
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_func_destroy(std::move(args));
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self, this](auto& fut) {
            self->_req_func_destroy = fut.get();
        });

}




void gpu_cuda_function::handle_call(auto args) {
    LOG(INFO) << "CALL handle call";
    using msg = ::service::compute::cuda::message::cuda_function::call;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    auto args_num = args->imms.args_num;
    auto grid = args->imms.grid;
    auto block = args->imms.block;

    auto raw_kernel_args = args->imms.kernel_args;

    void* kernel_args[args_num.get()];

    for (uint i = 0; i < args_num; i++) {
        auto info = (msg::kernel_arg_info*)raw_kernel_args;
        //LOG(INFO) << (void*)(uintptr_t)*(char*)info->value;
        kernel_args[i] = info->value;
        raw_kernel_args += info->size.get() + sizeof(msg::kernel_arg_info);

        //kernel_args[i] = (void*)raw_kernel_args;
        //raw_kernel_args += 8;
    }

    // dim3 dimGrid(grid);
    // dim3 dimBlock(block);

    // CUevent event;

    checkCudaErrors(cuLaunchKernel(_func, grid, 1, 1, 
        block, 1, 1, 
        0, 0, kernel_args, 0)); // 0 , stream , args, 0
    // checkCudaErrors(cuLaunchKernel(_func, dimGrid.x, dimGrid.y, dimGrid.z, 
    //                dimBlock.x, dimBlock.y, dimBlock.z, 
    //                0, 0, kernel_args, 0)); // 0 , stream , args, 0



    ch->make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
}


/*
 *  Destroy a cuda_function, revoke all of its caps
 */
void gpu_cuda_function::handle_func_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::message::cuda_function::func_destroy;

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

    // memory_free(base);

    DVLOG(logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->_req_call)
        .then([ch, self](auto& fut) {
            fut.get();
            LOG(INFO) << "Revoke _req_cumemalloc";
            return ch->revoke(self->_req_func_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DLOG(INFO) << "cuda function destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

