#include "srv_context.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/wire/error.hpp>



using namespace fractos;
using namespace ::test;
// using namespace impl;
#define MAX_IO_SIZE    (1024 * 1024 * 16)   

gpu_Context::gpu_Context(fractos::wire::endian::uint32_t value, CUdevice& device) {
    //fork();
    _id = value;
    _destroyed = false;   

    CUcontext ctx;
    checkCudaErrors(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, device)); //(unsigned int)value)); // 
    _ctx = ctx;
}

std::shared_ptr<gpu_Context> gpu_Context::factory(fractos::wire::endian::uint32_t value, CUdevice& device){

    auto res = std::shared_ptr<gpu_Context>(new gpu_Context(value, device));
    res->_self = res;
    return res;
}

gpu_Context::~gpu_Context() {
    checkCudaErrors(cuCtxDestroy(_ctx));
}


char* gpu_Context::allocate_memory(size_t size, CUcontext& context) {

    char* addr = nullptr;
    checkCudaErrors(cuCtxSetCurrent(context));
    CUdeviceptr d_A;
   
    // Allocate memory on the device
    checkCudaErrors(cuMemAlloc(&d_A, size));

 
    addr = (char*)d_A;
    //checkCudaErrors(cuCtxPopCurrent(nullptr));
    return addr;
}



void gpu_Context::context_synchronize() {
    checkCudaErrors(cuCtxSynchronize());
}

void gpu_Context::context_destroy(CUcontext& context) {
    checkCudaErrors(cuCtxDestroy(context));
}



/*
 *  Make handlers for a Context's caps
 */
core::future<void> gpu_Context::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Context;

    auto self = _self;


    return ch->make_request_builder<msg_base::make_memory::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            VLOG(fractos::logging::SERVICE) << "In register_service context handler";
            self->handle_memory(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_memory = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_memory"; 
            return ch->make_request_builder<msg_base::make_module_data::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_module_data(std::move(args)); // file
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_module_data = fut.get(); // file
            VLOG(fractos::logging::SERVICE) << "SET req_module_data"; // file
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
            VLOG(fractos::logging::SERVICE) << "SET req_synchronize"; 
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
            VLOG(fractos::logging::SERVICE) << "SET req_destroy context";
            self->_req_destroy = fut.get();
        });

}

/*
 *  Destroy a Context, revoke all of its caps
 */
void gpu_Context::handle_memory(auto args_) {
    VLOG(fractos::logging::SERVICE) << "CALL handle handle_memory";
    std::shared_ptr<typename decltype(args_)::element_type> args(std::move(args_));
    using msg = ::service::compute::cuda::wire::Context::make_memory;
    
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
        // VLOG(fractos::logging::SERVICE) << "cuda device addr: " << (void*)base;
        VLOG(fractos::logging::SERVICE) << "mem size is: " << size;


        auto dev_mem = std::shared_ptr<gpu_Memory>(gpu_Memory::factory(size, _ctx));
        

        dev_mem->_memory = fut.get();
        dev_mem->base = (char*)base;
        dev_mem->_mr = mr;


        dev_mem->register_methods(ch)
            .then([this, ch, self, dev_mem, size, args ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
                fut.get();
                _dev_mem = dev_mem;

                // VLOG(fractos::logging::SERVICE) << "BACKEND memory size is " << dev_mem->_memory.get_size();

                ch->make_request_builder<msg::response>(args->caps.continuation)
                    .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                    .set_imm(&msg::response::imms::address, dev_mem->_memory.get_addr())
                    .set_cap(&msg::response::caps::memory, dev_mem->_memory)
                    .set_cap(&msg::response::caps::destroy, dev_mem->_req_destroy)
                    .on_channel()
                    .invoke()
                    .as_callback();
                })
            .as_callback();
        })
        .as_callback();



}

// not in use.
void gpu_Context::handle_module_file(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle_module_file";

    using msg = ::service::compute::cuda::wire::Context::make_module_file;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
        return;
    }
    
    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    std::string file_name = args->imms.file_name;

    if (not args->has_exactly_args()) { // file_name
        if (not args->has_exactly_imms()) {
            if (args->imms_size() == 8 + file_name.size()) {
                VLOG(fractos::logging::SERVICE) << "got imms length : " << file_name.size(); // char file_name[] in msg
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
    VLOG(fractos::logging::SERVICE) << "module name is: " << file_name;


    auto mod = std::shared_ptr<gpu_Module>(gpu_Module::factory(file_name, _ctx));


    mod->register_methods(ch)
        .then([this, ch, self, mod, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _mod = mod;

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::get_function, mod->_req_get_func)
                .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
            })
        .as_callback();



}


void gpu_Context::handle_module_data(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle_module_data";

    using msg = ::service::compute::cuda::wire::Context::make_module_data;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
        return;
    }
    
    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    std::string file_name = args->imms.file_name;

    if (not args->has_exactly_args()) { // file_name
        if (not args->has_exactly_imms()) {
            if (args->imms_size() == 8 + file_name.size()) {
                VLOG(fractos::logging::SERVICE) << "got imms length : " << file_name.size(); // char file_name[] in msg
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
    VLOG(fractos::logging::SERVICE) << "module name is: " << file_name;

    auto size = args->caps.cuda_file.get_size();
    char* buffer = (char*)malloc(size);
    auto copied_mem = ch->make_memory(buffer, size).get();
    ch->copy(args->caps.cuda_file, copied_mem).get();


    auto mod = std::shared_ptr<gpu_Module>(gpu_Module::factory(file_name, _ctx, buffer, size));


    mod->register_methods(ch)
        .then([this, ch, self, mod, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _mod = mod;

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::get_function, mod->_req_get_func)
                .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
            })
        .as_callback();



}


void gpu_Context::handle_synchronize(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle synchronize";
    using msg = ::service::compute::cuda::wire::Context::synchronize;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();

    context_synchronize();

    ch->make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();
}



/*
 *  Destroy a Context, revoke all of its caps
 */
void gpu_Context::handle_destroy(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Context::destroy;

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

    VLOG(fractos::logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->_req_memory)
        .then([ch, self](auto& fut) {
                  fut.get();
                  VLOG(fractos::logging::SERVICE) << "Revoke _req_memory";
                  return ch->revoke(self->_req_module_data); // file
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_module_data"; // file
            return ch->revoke(self->_req_synchronize);
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_synchronize";
            return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Virtual context destroyed";
            this->_destroyed = true;
            ch->make_request_builder<msg::response>(args->caps.continuation) // response
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .on_channel()
                .invoke()
                .as_callback();
        })
    .as_callback();

}

