#include "srv_module.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>


using namespace fractos;
using namespace ::test;
// using namespace impl;

gpu_cuda_module::gpu_cuda_module(std::string& name, CUcontext& ctx) {
    //fork();
    // 
    _name = name;  // std::string 
    _destroyed = false;
    _ctx = ctx;

    CUmodule module;
    checkCudaErrors(cuCtxSetCurrent(_ctx));
    checkCudaErrors(cuModuleLoad(&module, _name.c_str()));
    _module = module;

    LOG(INFO) << "load module :  name = " << _name;
   
}

std::shared_ptr<gpu_cuda_module> gpu_cuda_module::factory(std::string& name, CUcontext& ctx){
    auto res = std::shared_ptr<gpu_cuda_module>(new gpu_cuda_module(name, ctx));
    res->_self = res;
    return res;
}

gpu_cuda_module::~gpu_cuda_module() {
    // checkCudaErrors(cuCtxDestroy(context));
    
}


void gpu_cuda_module::module_unload() // current
{
    // checkCudaErrors(cuCtxSetCurrent(_ctx));
    LOG(INFO) << "Unload module :  name = " << _name;
    checkCudaErrors(cuModuleUnload(_module));
    // checkCudaErrors(cuCtxDestroy(context));
}
/*
 *  Make handlers for a cuda_module's caps
 */
core::future<void> gpu_cuda_module::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_module;

    auto self = _self;


    return ch->make_request_builder<msg_base::get_cuda_function::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_get_function(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_get_func = fut.get();
            LOG(INFO) << "SET req_get_func"; // virtua
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
            self->_req_destroy = fut.get();
        });

}

void gpu_cuda_module::handle_get_function(auto args) {
    LOG(INFO) << "CALL handle_get_function";

    using msg = ::service::compute::cuda::message::cuda_module::get_cuda_function;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
        return;
    }
    
    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    std::string func_name = args->imms.func_name;

    if (not args->has_exactly_args()) { // file_name
        if (not args->has_exactly_imms()) {
            if (args->imms_size() == 8 + func_name.size()) {
                LOG(INFO) << "got imms length : " << func_name.size(); // char file_name[] in msg
            } else {
                LOG(ERROR) << "got error imms";
                ch->make_request_builder<msg::response>(args->caps.continuation)
                    .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
                    .on_channel()
                    .invoke()
                    .as_callback();
                return;
            }
        }
        else
        {
            LOG(ERROR) << "got error caps";
            ch->make_request_builder<msg::response>(args->caps.continuation)
                    .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
                    .on_channel()
                    .invoke()
                    .as_callback();
                return;
        }
    }

    auto self = _self;
    LOG(INFO) << "function name is: " << func_name;


    auto func = std::shared_ptr<gpu_cuda_function>(gpu_cuda_function::factory(func_name, _ctx, _module));


    func->register_methods(ch)
        .then([this, ch, self, func, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _func = func;

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::call, func->_req_call)
                .set_cap(&msg::response::caps::func_destroy, func->_req_func_destroy)
                .on_channel()
                .invoke()
                .as_callback();
            })
        .as_callback();
    
    // ch->make_request_builder<msg::response>(args->caps.continuation)
    //     .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
    //     .on_channel()
    //     .invoke()
    //     .as_callback();



}


/*
 *  Destroy a cuda_module, revoke all of its caps
 */
void gpu_cuda_module::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::message::cuda_module::destroy;

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

    module_unload();

    DVLOG(logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->_req_get_func)
        .then([ch, self](auto& fut) {
                  fut.get();
                  LOG(INFO) << "Revoke _req_get_func";
                  return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DLOG(INFO) << "cuda module destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

