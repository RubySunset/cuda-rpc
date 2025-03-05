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
#define MAX_IO_SIZE    (1024 * 1024 * 16)   

gpu_cuda_context::gpu_cuda_context(fractos::wire::endian::uint32_t value, CUdevice& device) {
    //fork();
    _id = value;
    _destroyed = false;   

    CUcontext ctx;
    checkCudaErrors(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, device)); //(unsigned int)value)); // 
    _ctx = ctx;
}

std::shared_ptr<gpu_cuda_context> gpu_cuda_context::factory(fractos::wire::endian::uint32_t value, CUdevice& device){

    auto res = std::shared_ptr<gpu_cuda_context>(new gpu_cuda_context(value, device));
    res->_self = res;
    return res;
}

gpu_cuda_context::~gpu_cuda_context() {
    checkCudaErrors(cuCtxDestroy(_ctx));
}


char* gpu_cuda_context::allocate_memory(size_t size, CUcontext& context) {

    char* addr = nullptr;
    checkCudaErrors(cuCtxSetCurrent(context));
    CUdeviceptr d_A;
   
    // Allocate memory on the device
    checkCudaErrors(cuMemAlloc(&d_A, size));

 
    addr = (char*)d_A;
    //checkCudaErrors(cuCtxPopCurrent(nullptr));
    return addr;
}



void gpu_cuda_context::context_synchronize() {
    checkCudaErrors(cuCtxSynchronize());
}

void gpu_cuda_context::context_destroy(CUcontext& context) {
    checkCudaErrors(cuCtxDestroy(context));
}



/*
 *  Make handlers for a cuda_context's caps
 */
core::future<void> gpu_cuda_context::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::message::cuda_context;

    auto self = _self;


    return ch->make_request_builder<msg_base::make_cumemalloc::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            LOG(INFO) << "In register_service context handler";
            self->handle_cumemalloc(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_cumemalloc = fut.get();
            LOG(INFO) << "SET req_cumemalloc"; // virtua
            return ch->make_request_builder<msg_base::make_module_file::request>(
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_module_file(std::move(args));
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_module_file = fut.get();
            LOG(INFO) << "SET req_module_file"; // virtua
            return ch->make_request_builder<msg_base::synchronize::request>(
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_synchronize(std::move(args));
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_synchronize = fut.get();
            LOG(INFO) << "SET req_synchronize"; // virtua
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
void gpu_cuda_context::handle_cumemalloc(auto args_) {
    LOG(INFO) << "CALL handle handle_cumemalloc";
    std::shared_ptr<typename decltype(args_)::element_type> args(std::move(args_));
    using msg = ::service::compute::cuda::message::cuda_context::make_cumemalloc;
    
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
    auto self = _self;
    std::size_t size = args->imms.size; // uint64_t MAX_IO_SIZE;

    
    // char *base;
    // if (posix_memalign((void **) &base, 4096, MAX_IO_SIZE) != 0) {
    //     throw std::runtime_error("Failed to allocate I/O buffer for worker.");
    // }

    char* base = allocate_memory(size , _ctx);//, context);



    auto mr_ = ch->make_memory_region(base, size, core::memory_region::translation_type::PIN);
    std::shared_ptr<typename decltype(mr_)::element_type> mr(std::move(mr_)); // element_type??


    ch->make_memory(base, size, *mr)
        .then([ch, args, size, this, base, mr](auto& fut) {
            // auto mem = fut.get();

        

        auto self = _self; // lock()
        // LOG(INFO) << "cuda device addr: " << (void*)base;
        LOG(INFO) << "mem size is: " << size;


        auto dev_mem = std::shared_ptr<gpu_cuda_memory>(gpu_cuda_memory::factory(size, _ctx));
        

        dev_mem->_memory = fut.get();
        dev_mem->base = (char*)base;
        dev_mem->_mr = mr;


        dev_mem->register_methods(ch)
            .then([this, ch, self, dev_mem, size, args ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
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
        })
        .as_callback();



}

void gpu_cuda_context::handle_module_file(auto args) {
    LOG(INFO) << "CALL handle_module_file";

    using msg = ::service::compute::cuda::message::cuda_context::make_module_file;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
        return;
    }
    
    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    std::string func_name = args->imms.func_name;

    if (not args->has_exactly_args()) { // func_name
        if (not args->has_exactly_imms()) {
            if (args->imms_size() == 8 + func_name.size()) {
                LOG(INFO) << "got imms length : " << func_name.size(); // char func_name[] in msg
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
    LOG(INFO) << "module name is: " << func_name;


    auto mod = std::shared_ptr<gpu_cuda_module>(gpu_cuda_module::factory(func_name, _ctx));


    mod->register_methods(ch)
        .then([this, ch, self, mod, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _mod = mod;

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                // .set_imm(&msg::response::imms::address, dev_mem->_memory.get_addr())
                // .set_cap(&msg::response::caps::memory, dev_mem->_memory)
                .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
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




void gpu_cuda_context::handle_synchronize(auto args) {
    LOG(INFO) << "CALL handle synchronize";
    using msg = ::service::compute::cuda::message::cuda_context::synchronize;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    // context_synchronize();

    ch->make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .on_channel()
        .invoke()
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

    context_destroy(_ctx);

    LOG(INFO) << "Revoke destroy";

    ch->revoke(self->_req_cumemalloc)
        .then([ch, self](auto& fut) {
                  fut.get();
                  LOG(INFO) << "Revoke _req_cumemalloc";
                  return ch->revoke(self->_req_module_file);
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            LOG(INFO) << "Revoke _req_module_file";
            return ch->revoke(self->_req_synchronize);
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            LOG(INFO) << "Revoke _req_synchronize";
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

}

