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
#include "./function.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Module;
using namespace fractos;


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


impl::Module::Module(CUcontext& ctx, std::shared_ptr<char[]>& buffer, size_t size, std::weak_ptr<Context> vctx) {
    //fork();
    // 

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

}


impl::Module::Module(std::string& name, CUcontext& ctx) {
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

std::shared_ptr<impl::Module> impl::Module::factory(std::string& name, CUcontext& ctx){
    auto res = std::shared_ptr<Module>(new Module(name, ctx));
    res->_self = res;
    return res;
}

std::shared_ptr<impl::Module> impl::Module::factory(CUcontext& ctx, std::shared_ptr<char[]>& buffer, size_t size, std::weak_ptr<Context> vctx){
    auto res = std::shared_ptr<Module>(new Module(ctx, buffer, size, vctx));
    res->_self = res;
    return res;
}



impl::Module::~Module() {
    // checkCudaErrors(cuCtxDestroy(context));
    
}


void impl::Module::module_unload() // current
{
    checkCudaErrors(cuCtxSetCurrent(_ctx));
    VLOG(fractos::logging::SERVICE) << "Unload module :  name = " << _name;
    checkCudaErrors(cuModuleUnload(_module));
    // checkCudaErrors(cuCtxDestroy(context));
}
/*
 *  Make handlers for a Module's caps
 */
core::future<void> impl::Module::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Module;

    auto self = _self;


    return ch->make_request_builder<msg_base::generic::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self](auto& fut) {
            self->_req_generic = fut.get();
        })
        .then([ch, self](auto& fut) {
            fut.get();
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

CUmodule
impl::Module::get_remote_cumodule() const
{
    return (CUmodule)this;
}

void
impl::Module::handle_generic(auto ch, auto args)
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

#define CASE_HANDLE(NAME, name)                                         \
    case srv_wire_msg::OP_ ## NAME:                                      \
        handle_ ## name(ch, reinterpreted.template operator()<srv_wire_msg:: name ::request>(std::move(args))); \
        break;

    switch (opcode) {
    CASE_HANDLE(GET_GLOBAL, get_global);
    CASE_HANDLE(GET_FUNCTION, get_function);
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
impl::Module::handle_get_global(auto ch, auto args)
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
    auto cuerror = CUDA_SUCCESS;
    CUdeviceptr dptr = 0;

    cuerror = cuCtxSetCurrent(_ctx);
    if (cuerror != CUDA_SUCCESS) {
        goto out;
    }

    cuerror = cuModuleGetGlobal(&dptr, nullptr, _module, name.c_str());

out:

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror)
        << " dptr=" << (void*)dptr;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .set_imm(&msg::response::imms::dptr, dptr)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Module::handle_get_function(auto ch, auto args)
{
    METHOD(get_function);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_CAPS_EXACT(reqb_cont);
    CHECK_IMMS_ALL(reqb_cont);
    CHECK_ARGS_COND(reqb_cont,
                    args->imms_size() == (sizeof(msg::request::imms) + args->imms.name_size));

    auto self = _self;
    std::string name(args->imms.name, args->imms.name_size);

    make_function(ch, _vctx.lock(), self, name)
        .then([this, self, ch, args=std::move(args)](auto& fut) {
            auto [error, cuerror, func] = fut.get();

            _func = func;

            auto args_size_offset = offsetof(msg::response::imms, arg_size);
            std::vector<wire::endian::uint64_t> args_size;
            for (auto arg: func->args_size) {
                args_size.push_back(arg);
            }

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::cufunction, (uint64_t)func->get_remote_cufunction())
                .set_imm(&msg::response::imms::nargs, func->args_size.size())
                .set_imm(args_size_offset, args_size.data(), sizeof(uint64_t) * func->args_size.size())
                .set_cap(&msg::response::caps::generic, func->req_generic)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
            })
        .as_callback_log_ignore_continuation_error();
}


/*
 *  Destroy a Module, revoke all of its caps
 */
void impl::Module::handle_destroy(auto args) {
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

    ch->revoke(self->_req_generic)
        .then([ch, self](auto& fut) {
            fut.get();
            return ch->revoke(self->_req_destroy);
        })
        .unwrap()
        .then([ch, this, self, args=std::move(args)](auto& fut) {
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
impl::to_string(const Module& obj)
{
    std::stringstream ss;
    ss << "Module(" << &obj << ")";
    return ss.str();
}
