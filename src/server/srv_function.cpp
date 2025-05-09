#include <cuda_runtime.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "common.hpp"
#include "srv_context.hpp"
// #include "srv_function.hpp"


using namespace fractos;
namespace srv = fractos::service::compute::cuda;
using namespace ::test;

#define checkCudaErrors_lo(err)  handleError(err, __FILE__, __LINE__)

void handleError(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        LOG(INFO) << "CUDA Driver API error = " << err
                    << " from file <" << file << ">, line " << line << ".\n";
        // exit(-1);
    }
    LOG(INFO) << "CUDA Driver API SUCCESS from file <" << file << ">, line " << line << ".\n";
}



gpu_Function::gpu_Function(std::string func_name, CUcontext& ctx, CUmodule& mod, std::weak_ptr<test::gpu_Context> vctx) {

    _name = func_name;
    _destroyed = false;
    _ctx = ctx;
    _mod = mod;
    _vctx = vctx;

    checkCudaErrors_lo(cuCtxSetCurrent(_ctx));

    CUfunction function;
    checkCudaErrors_lo(cuModuleGetFunction(&function, _mod, func_name.c_str()));
    _func = function;

    _args_total_size = 0;
    for (size_t i = 0; true; i++) {
        size_t offset, size;
        auto res = cuFuncGetParamInfo(_func, i, &offset, &size);
        if (res == CUDA_SUCCESS) {
            _args_total_size += size;
            _args_size.push_back(size);
        } else if (res == CUDA_ERROR_INVALID_VALUE) {
            break;
        } else {
            LOG(FATAL) << "unexpected error: " << res;
        }
    }
}

std::shared_ptr<gpu_Function> gpu_Function::factory(std::string func_name, CUcontext& ctx, CUmodule& mod
                                                ,std::weak_ptr<test::gpu_Context> vctx){
    auto res = std::shared_ptr<gpu_Function>(new gpu_Function(func_name, ctx, mod, vctx));
    res->_self = res;
    return res;
}

gpu_Function::~gpu_Function() {
    // checkCudaErrors(cuCtxDestroy(context));
}

/*
 *  Make handlers for a Function's caps
 */
core::future<void> gpu_Function::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Function;

    auto self = _self;


    return ch->make_request_builder<msg_base::generic::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self, ch](auto& fut) {
            self->_req_generic = fut.get();
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

void
gpu_Function::handle_generic(auto ch, auto args)
{
    METHOD(Function, generic);

    auto opcode = srv_wire::OP_INVALID;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG_OP(method)
            << " [error] request without continuation, ignoring";
        return;
    } else if (args->has_imm(&msg::request::imms::opcode)) {
        opcode = static_cast<srv_wire::generic_opcode>(args->imms.opcode.get());
    }

    auto reinterpreted = []<class T>(auto args) {
        using ptr = core::receive_args<T>;
        return std::unique_ptr<ptr>(reinterpret_cast<ptr*>(args.release()));
    };

#define HANDLE(name) \
    handle_ ## name(ch, reinterpreted.template operator()<srv_wire:: name ::request>(std::move(args)))

    switch (opcode) {
    case srv_wire::OP_LAUNCH:
        HANDLE(launch);
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
gpu_Function::handle_launch(auto ch, auto args)
{
    METHOD(Function, launch);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_CAPS_EXACT();
    CHECK_IMMS_ALL();
    CHECK_ARGS_COND(args->imms_size() == (sizeof(msg::request::imms) + _args_total_size));

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    auto ctx_ptr = _vctx.lock();
    CHECK(ctx_ptr);
    cuerror = cuCtxSetCurrent(ctx_ptr->_ctx);
    if (cuerror != CUDA_SUCCESS) {
        goto out;
    }

    {
        dim3 dimGrid(args->imms.grid_x, args->imms.grid_y, args->imms.grid_z);
        dim3 dimBlock(args->imms.block_x, args->imms.block_y, args->imms.block_z);

        std::vector<const void*> kernel_args;
        const char* args_ptr = args->imms.kernel_args;
        for (size_t i = 0; i < _args_size.size(); i++) {
            kernel_args.push_back(args_ptr);
            args_ptr += _args_size[i];
        }

        int stream_id = args->imms.stream_id;
        if (stream_id) {
            LOG_FIRST_N(WARNING, 1) << "TODO: add a proper API to query/get stream_ids";
            LOG_FIRST_N(WARNING, 1) << "TODO: return error when stream_id is incorrect";
            auto _vstream = ctx_ptr->getVStreamMap().at(stream_id);
            cuerror = cuLaunchKernel(_func, dimGrid.x, dimGrid.y, dimGrid.z,
                                     dimBlock.x, dimBlock.y, dimBlock.z,
                                     0, _vstream->getCUStream(),
                                     (void**)kernel_args.data(), 0);
        } else {
            cuerror = cuLaunchKernel(_func, dimGrid.x, dimGrid.y, dimGrid.z,
                                     dimBlock.x, dimBlock.y, dimBlock.z,
                                     0, 0,
                                     (void**)kernel_args.data(), 0);
        }
    }

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << cudaGetErrorString((cudaError)cuerror);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}

/*
 *  Destroy a Function, revoke all of its caps
 */
void gpu_Function::handle_func_destroy(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Function::func_destroy;

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

    ch->revoke(self->_req_generic)
        .then([ch, self](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Revoke _req_memory";
            return ch->revoke(self->_req_func_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            DVLOG(fractos::logging::SERVICE) << "cuda function destroyed";
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
test::to_string(const gpu_Function& obj)
{
    std::stringstream ss;
    ss << "Function(" << &obj << ")";
    return ss.str();
}
