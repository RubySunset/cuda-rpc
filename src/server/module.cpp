#include <cuda.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <fstream>
#include <glog/logging.h>
#include <iostream>
#include <pthread.h>

#include "common.hpp"
#include "./module.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Module;
using namespace fractos;
using namespace ::test;
using namespace impl;


#define checkCudaErrors_lo(err)  handleError_lo(err, __FILE__, __LINE__)

void handleError_lo(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        LOG(INFO) << "CUDA Driver API error = " << err
                    << " from file <" << file << ">, line " << line << ".\n";
        // exit(-1);
    }
    LOG(INFO) << "CUDA Driver API SUCCESS from file <" << file << ">, line " << line << ".\n";
}

void check_memory()
{

    size_t freeMem;
    size_t totalMem;

    cuMemGetInfo(&freeMem, &totalMem);

    // Print memory info
    LOG(INFO) << "Free memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    LOG(INFO) << "Total memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // const char* command = "nvidia-smi -q | grep -A 3 \"BAR1 Memory Usage\"";
    // std::array<char, 128> buffer;
    // std::string result;

    // // Open a pipe to run the command
    // std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command, "r"), pclose);
    // if (!pipe) {
    //     LOG(ERROR) << "Failed to run command." << std::endl;
    //     // exit(-1);
    // }

    // // Read the output of the command
    // while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    //     result += buffer.data();
    // }

    // // Write the result to a file
    // std::ofstream outFile("bar1_memory_usage.txt",  std::ios::binary | std::ios::app);
    // if (outFile.is_open()) {
    //     outFile << result;
    //     outFile.close();
    //     LOG(INFO) << "Output written to bar1_memory_usage.txt" << std::endl;
    // } else {
    //     LOG(ERROR) << "Failed to open file for writing." << std::endl;
    // }

}


gpu_Module::gpu_Module(uint64_t module_id, CUcontext& ctx, std::shared_ptr<char[]>& buffer, size_t size, std::weak_ptr<Context> vctx) {
    //fork();
    // 

    _id = module_id;  // std::string 
    _destroyed = false;
    _ctx = ctx;
    _vctx = vctx;
    _data = buffer;

    checkCudaErrors_lo(cuCtxSetCurrent(_ctx));
    // checkCudaErrors_lo(cuCtxSynchronize());

    check_memory();

    CUmodule module2;
    checkCudaErrors_lo(cuModuleLoadData(&module2, _data.get()));
    _module = module2;

    VLOG(fractos::logging::SERVICE) << "load module :  id = " << _id;
   
}


gpu_Module::gpu_Module(std::string& name, CUcontext& ctx) {
    //fork();
    // 
    _name = name;  // std::string 
    _destroyed = false;
    _ctx = ctx;

    CUmodule module;
    checkCudaErrors(cuCtxSetCurrent(_ctx));
    
    checkCudaErrors(cuModuleLoad(&module, _name.c_str()));
    _module = module;

    VLOG(fractos::logging::SERVICE) << "load module :  name = " << _name;
   
}

std::shared_ptr<gpu_Module> gpu_Module::factory(std::string& name, CUcontext& ctx){
    auto res = std::shared_ptr<gpu_Module>(new gpu_Module(name, ctx));
    res->_self = res;
    return res;
}

std::shared_ptr<gpu_Module> gpu_Module::factory(uint64_t module_id, CUcontext& ctx, std::shared_ptr<char[]>& buffer, size_t size, std::weak_ptr<Context> vctx){
    auto res = std::shared_ptr<gpu_Module>(new gpu_Module(module_id, ctx, buffer, size, vctx));
    res->_self = res;
    return res;
}



gpu_Module::~gpu_Module() {
    // checkCudaErrors(cuCtxDestroy(context));
    
}


void gpu_Module::module_unload() // current
{
    checkCudaErrors(cuCtxSetCurrent(_ctx));
    VLOG(fractos::logging::SERVICE) << "Unload module :  name = " << _name;
    checkCudaErrors(cuModuleUnload(_module));
    // checkCudaErrors(cuCtxDestroy(context));
}
/*
 *  Make handlers for a Module's caps
 */
core::future<void> gpu_Module::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Module;

    auto self = _self;


    return ch->make_request_builder<msg_base::get_function::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_get_function(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_get_func = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_get_func"; // virtua
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
gpu_Module::handle_generic(auto ch, auto args)
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

#define HANDLE(name) \
    handle_ ## name(ch, reinterpreted.template operator()<srv_wire_msg:: name ::request>(std::move(args)))

    switch (opcode) {
    case srv::wire::Module::OP_GET_GLOBAL:
        HANDLE(get_global);
        break;

    default:
        LOG_OP(method)
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
gpu_Module::handle_get_global(auto ch, auto args)
{
    METHOD(get_global);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_CAPS_EXACT(reqb_cont);
    CHECK_IMMS_ALL(reqb_cont);
    CHECK_ARGS_COND(reqb_cont,
                    args->imms_size() == (sizeof(msg::request::imms) + args->imms.name_size));

    std::string name(args->imms.name, args->imms.name_size);

    auto error = wire::ERR_SUCCESS;
    auto cuerr = CUDA_SUCCESS;
    CUdeviceptr dptr = 0;

    cuerr = cuCtxSetCurrent(_ctx);
    if (cuerr != CUDA_SUCCESS) {
        goto out;
    }

    cuerr = cuModuleGetGlobal(&dptr, nullptr, _module, name.c_str());

out:
    if (cuerr != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " dptr=" << (void*)dptr;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::dptr, dptr)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void gpu_Module::handle_get_function(auto args) {
    auto t_start = std::chrono::high_resolution_clock::now();
    VLOG(fractos::logging::SERVICE) << "CALL handle_get_function";

    using msg = ::service::compute::cuda::wire::Module::get_function;
    
    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "got request without continuation, ignoring";
        return;
    }
    
    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    std::string func_name = args->imms.func_name;

    if (not args->has_exactly_args()) { // file_name
        if (not args->has_exactly_imms()) {
            if (args->imms_size() == 8 + func_name.size()) {
                VLOG(fractos::logging::SERVICE) << "got imms length : " << func_name.size(); // char file_name[] in msg
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
    VLOG(fractos::logging::SERVICE) << "function name is: " << func_name;


    // std::shared_ptr<Context> _vctx = _vctx.lock();

    auto [err, func] = impl::make_function(_vctx.lock(), _module, func_name);
    CHECK(err == CUDA_SUCCESS) << "TODO: return error";

    func->register_methods(ch)
        .then([this, ch, self, func, args=std::move(args) ](auto& fut) { //, args=std::move(args),  mr_=std::move(mr_)
            fut.get();
            _func = func;

            auto args_size_offset = offsetof(msg::response::imms, arg_size);
            std::vector<wire::endian::uint64_t> args_size;
            for (auto arg: func->args_size) {
                args_size.push_back(arg);
            }

            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_imm(&msg::response::imms::nargs, func->args_size.size())
                .set_imm(args_size_offset, args_size.data(), sizeof(uint64_t) * func->args_size.size())
                .set_cap(&msg::response::caps::generic, func->req_generic)
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

    auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start);
    LOG(INFO) << "time for get function server: " << t_usec.count() << std::endl;
}


/*
 *  Destroy a Module, revoke all of its caps
 */
void gpu_Module::handle_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Module::destroy;

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
                  VLOG(fractos::logging::SERVICE) << "Revoke _req_get_func";
                  return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DVLOG(fractos::logging::SERVICE) << "cuda module destroyed";
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
test::to_string(const gpu_Module& obj)
{
    std::stringstream ss;
    ss << "Module(" << &obj << ")";
    return ss.str();
}
