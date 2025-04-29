#include "srv_context.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <fstream>

#include "./common.hpp"


using namespace fractos;
namespace srv = fractos::service::compute::cuda;
using namespace ::test;
// using namespace impl;
#define MAX_IO_SIZE    (1024 * 1024 * 16)   

#define checkCudaErrors_lo(err)  handleError_llo(err, __FILE__, __LINE__)

void handleError_llo(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        LOG(INFO) << "CUDA Driver API error = " << err
                    << " from file <" << file << ">, line " << line << ".\n";
        // exit(-1);
    }
    LOG(INFO) << "CUDA Driver API SUCCESS from file <" << file << ">, line " << line << ".\n";
}



gpu_Context::gpu_Context(fractos::wire::endian::uint32_t value, CUdevice device) {
    //fork();
    _id = value;
    _destroyed = false;   

    CUcontext ctx;
    checkCudaErrors(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, device)); //(unsigned int)value)); // 
    _ctx = ctx;
}

std::shared_ptr<gpu_Context> gpu_Context::factory(fractos::wire::endian::uint32_t value, CUdevice device){

    auto res = std::shared_ptr<gpu_Context>(new gpu_Context(value, device));
    res->_self = res;
    return res;
}

gpu_Context::~gpu_Context() {
    checkCudaErrors(cuCtxDestroy(_ctx));
}

const std::unordered_map<int, std::shared_ptr<gpu_Stream>>& gpu_Context::getVStreamMap() const {
    return _vstream_map;
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
        ch->get_default_endpoint(), // resv msg
        [self](auto ch, auto args) {
            VLOG(fractos::logging::SERVICE) << "In register_service context handler";
            self->handle_memory(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_memory = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_memory_rpc_test"; 
            return ch->make_request_builder<msg_base::make_memory_rpc_test::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_memory_rpc_test(std::move(args)); // file
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_memory_rpc_test = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_memory"; 
            return ch->make_request_builder<msg_base::make_stream::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_stream(std::move(args)); // file
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_stream = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_stream"; 
            return ch->make_request_builder<msg_base::make_event::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_event(std::move(args)); // file
                })
                .on_channel()
                .make_request();
            })
        .unwrap()
        .then([ch, self](auto& fut) {
            self->_req_event = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_event"; 
            return ch->make_request_builder<msg_base::make_module_data::request>( // file
                ch->get_default_endpoint(), 
                [self](auto ch, auto args) {
                    
                    self->handle_module_data(std::move(args)); // data
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

            return ch->make_request_builder<msg_base::generic::request>(
                ch->get_default_endpoint(),
                [self](auto ch, auto args) {
                    self->handle_generic(ch, std::move(args));
                })
                .on_channel()
                .make_request();
        })
        .unwrap()
        .then([ch, self, this](auto& fut) {
            self->_req_generic = fut.get();
        });
}
void
gpu_Context::handle_generic(auto ch, auto args)
{
    static const std::string method = "handle_generic";
    using msg = srv::wire::Context::generic;

    auto opcode = srv::wire::Context::OP_INVALID;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG_OP(method)
            << " [error] request without continuation, ignoring";
        return;
    } else if (args->has_imm(&msg::request::imms::opcode)) {
        opcode = static_cast<srv::wire::Context::generic_opcode>(args->imms.opcode.get());
    }

    auto reinterpreted = []<class T>(auto args) {
        using ptr = core::receive_args<T>;
        return std::unique_ptr<ptr>(reinterpret_cast<ptr*>(args.release()));
    };

#define HANDLE(name) \
    handle_ ## name(ch, reinterpreted.template operator()<srv::wire::Context:: name ::request>(std::move(args)))

    switch (opcode) {
    case srv::wire::Context::OP_GET_API_VERSION:
        HANDLE(get_api_version);
        break;

    default:
        LOG_OP(method)
            << " [error] invalid opcode";
        ch->template make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
        break;
    }

#undef HANDLE
}

void
gpu_Context::handle_get_api_version(auto ch, auto args)
{
    METHOD(Context, get_api_version);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    unsigned int version;
    auto res = cuCtxGetApiVersion(_ctx, &version);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " version=" << version;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::version, version)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}

/*
 *  Destroy a Context, revoke all of its caps
 */
void gpu_Context::handle_memory(auto args_) {
    using clock = std::chrono::high_resolution_clock;
    std::chrono::microseconds t_usec;
    auto t_start = clock::now();
    // auto t_end = clock::now();
    // auto t_temp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    // LOG(INFO) << "time for make_memory server start: " << t_temp << std::endl;




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


    t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
    LOG(INFO)  << "time for make_memory CUDA API server: "<< t_usec.count() << std::endl;



    auto mr_ = ch->make_memory_region(base, size, core::memory_region::translation_type::PIN); // PIN
    std::shared_ptr<typename decltype(mr_)::element_type> mr(std::move(mr_)); // element_type??

    auto temp = t_usec.count();
    t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
    LOG(INFO)  << "time for make_memory make_memory_region server: "<< t_usec.count() - temp << std::endl;



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

                auto t_temp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                LOG(INFO)  << "time for make_memory make_memory server: "<< t_temp << std::endl;

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

    // temp = t_usec.count();
    // t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
    // LOG(INFO)  << "time for make_memory make_memory server: "<< t_usec.count() - temp << std::endl;

    t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
    LOG(INFO)  << "time for make_memory server: "<< t_usec.count() << std::endl;

}

void gpu_Context::handle_memory_rpc_test(auto args) {

    using clock = std::chrono::high_resolution_clock;
    std::chrono::microseconds t_usec;
    auto t_start = clock::now();

    using msg = ::service::compute::cuda::wire::Context::make_memory_rpc_test;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    
    LOG(INFO) << "Revoke test";

    ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
            .on_channel()
            .invoke()
            .as_callback();

        // return;

    t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
    LOG(INFO)  << "time for memory_rpc_test server: "<< t_usec.count() << std::endl;
}


void gpu_Context::handle_stream(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle_stream";
    using msg = ::service::compute::cuda::wire::Context::make_stream;
    
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

    unsigned int flag = args->imms.flags; // uint32_t
    int id = (int)args->imms.stream_id;

    auto self = _self; // lock()

    VLOG(fractos::logging::SERVICE) << "vstream flag is: " << (uint32_t)flag;
    LOG(INFO) << "vstream id is: " << id;

    auto stream = std::shared_ptr<gpu_Stream>(gpu_Stream::factory(flag, id, _ctx));

    stream->register_methods(ch)
        .then([this, ch, self, stream, args=std::move(args), flag, id](auto& fut) {
            fut.get();
            _stream = stream;
            _vstream_map.insert({id, stream});
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::synchronize, stream->_req_sync)
                .set_cap(&msg::response::caps::destroy, stream->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}


void gpu_Context::handle_event(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle_event";
    using msg = ::service::compute::cuda::wire::Context::make_event;
    
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

    unsigned int flag = args->imms.flags; // uint32_t

    auto self = _self; // lock()

    VLOG(fractos::logging::SERVICE) << "event flag is: " << (uint32_t)flag;

    auto event = std::shared_ptr<gpu_Event>(gpu_Event::factory(flag, _ctx));

    event->register_methods(ch)
        .then([this, ch, self, event, args=std::move(args), flag](auto& fut) {
            fut.get();
            _event = event;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                // .set_cap(&msg::response::caps::synchronize, event->_req_sync)
                .set_cap(&msg::response::caps::destroy, event->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}


// // not in use.
// void gpu_Context::handle_module_file(auto args) {
//     VLOG(fractos::logging::SERVICE) << "CALL handle_module_file";

//     using msg = ::service::compute::cuda::wire::Context::make_module_file;
    
//     if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
//         LOG(ERROR) << "got request without continuation, ignoring";
//         return;
//     }
    
//     std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
//     std::string file_name = args->imms.file_name;

//     if (not args->has_exactly_args()) { // file_name
//         if (not args->has_exactly_imms()) {
//             if (args->imms_size() == 8 + file_name.size()) {
//                 VLOG(fractos::logging::SERVICE) << "got imms length : " << file_name.size(); // char file_name[] in msg
//             } else {
//                 LOG(ERROR) << "got error imms";
//                 ch->make_request_builder<msg::response>(args->caps.continuation)
//                     .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
//                     .on_channel()
//                     .invoke()
//                     .as_callback();
//                 return;
//             }
//         }
//         else
//         {
//             LOG(ERROR) << "got error caps";
//             ch->make_request_builder<msg::response>(args->caps.continuation)
//                     .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
//                     .on_channel()
//                     .invoke()
//                     .as_callback();
//                 return;
//         }
//     }

//     auto self = _self;
//     VLOG(fractos::logging::SERVICE) << "module name is: " << file_name;


//     auto mod = std::shared_ptr<gpu_Module>(gpu_Module::factory(file_name, _ctx));


//     mod->register_methods(ch)
//         .then([this, ch, self, mod, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
//             fut.get();
//             _mod = mod;

//             ch->make_request_builder<msg::response>(args->caps.continuation)
//                 .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
//                 .set_cap(&msg::response::caps::get_function, mod->_req_get_func)
//                 .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
//                 .on_channel()
//                 .invoke()
//                 .as_callback();
//             })
//         .as_callback();



// }


void gpu_Context::handle_module_data(auto args) {
    auto t_start = std::chrono::high_resolution_clock::now();

    VLOG(fractos::logging::SERVICE) << "CALL handle_module_data";

    using msg = ::service::compute::cuda::wire::Context::make_module_data;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
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

    auto module_id = args->imms.module_id;

    auto self = _self;
    VLOG(fractos::logging::SERVICE) << "module id is: " << module_id;

    
    auto size = args->caps.cuda_file.get_size();

    std::shared_ptr<char> buffer(new char[size], std::default_delete<char[]>());
    // char* buffer = (char*)malloc(size);

    auto mr_lo = ch->make_memory_region(buffer.get(), size, core::memory_region::translation_type::PIN);

    auto copied_mem = ch->make_memory(buffer.get(), size, *mr_lo).get();
    ch->copy(args->caps.cuda_file, copied_mem).get();
    LOG(INFO) << "get cuda_file in memory";

    // std::ofstream outfile("module_code.ptx", std::ios::binary | std::ios::app);
    // if (outfile.is_open()) {
    //     outfile.write(buffer.get(), size);
    //     outfile.close();
    //     std::cout << "Buffer appended to module_code.ptx successfully." << std::endl;
    // } else {
    //     std::cerr << "Failed to open file for writing." << std::endl;
    // }

    // CUcontext newContext;
    // checkCudaErrors(cuCtxSetCurrent(_ctx));
    // CUmodule module;
    // checkCudaErrors_lo(cuModuleLoadData(&module, buffer.get()));
    
    if (buffer.get()[0] ) //buffer.get()[0] 
    {
        std::cout << "Buffer for ptx file is valid." << std::endl;
    }
    else{
        std::cerr << "ptx buffer is not valid for load" << std::endl;
    }

    auto mod = std::shared_ptr<gpu_Module>(gpu_Module::factory(module_id, _ctx, buffer, size, self));

    // auto mod = std::shared_ptr<gpu_Module>(gpu_Module::factory(file_name, _ctx));


    mod->register_methods(ch)
        .then([this, ch, self, mod, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _mod = mod;

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::get_function, mod->_req_get_func)
                .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
                .on_channel()
                .invoke() // op . then
                .as_callback();
            // auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start);
            // LOG(INFO) << "time for load module server: " << t_usec.count() << std::endl;
            })
        .as_callback();

    auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start);
    LOG(INFO) << "time for load module server: " << t_usec.count() << std::endl;
    
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
                  return ch->revoke(self->_req_memory_rpc_test); // file
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_memory";
            return ch->revoke(self->_req_stream); // file
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_memory";
            return ch->revoke(self->_req_event); // file
        })
        .unwrap()
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_stream";
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

std::string
test::to_string(const gpu_Context& obj)
{
    std::stringstream ss;
    ss << "Context(" << &obj << ")";
    return ss.str();
}
