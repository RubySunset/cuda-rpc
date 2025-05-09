// #include "srv_function.hpp"
#include "srv_context.hpp"
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

#include <cuda_runtime.h>


using namespace fractos;
using namespace ::test;
// using namespace impl;

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


    return ch->make_request_builder<msg_base::call::request>(
        ch->get_default_endpoint(), 
        [self](auto ch, auto args) {
            self->handle_call(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->_req_call = fut.get();
            VLOG(fractos::logging::SERVICE) << "SET req_call"; // virtua
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




void gpu_Function::handle_call(auto args) {

    auto t_start = std::chrono::high_resolution_clock::now();


    VLOG(fractos::logging::SERVICE) << "CALL handle call";
    using msg = ::service::compute::cuda::wire::Function::call;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps.continuation.get_channel();

    if (not args->has_exactly_caps() or
        args->imms_size() != args->imms_expected_size() + _args_total_size) {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();
        return;
    }

    auto ctx_ptr = _vctx.lock();
    CHECK(ctx_ptr);
    CHECK(cuCtxSetCurrent(ctx_ptr->_ctx) == CUDA_SUCCESS);

    auto grid_x = args->imms.grid_x;
    auto grid_y = args->imms.grid_y;
    auto grid_z = args->imms.grid_z;
    auto block_x = args->imms.block_x;
    auto block_y = args->imms.block_y;
    auto block_z = args->imms.block_z;

    std::vector<const void*> kernel_args;
    const char* args_ptr = args->imms.kernel_args;
    for (size_t i = 0; i < _args_size.size(); i++) {
        kernel_args.push_back(args_ptr);
        args_ptr += _args_size[i];
    }


    dim3 dimGrid(grid_x, grid_y, grid_z);
    dim3 dimBlock(block_x, block_y, block_z);

    // CUevent event;
    // CUstream stream;

    LOG(INFO) << "STREAM ID " << (int)args->imms.stream_id;
    if ((int)args->imms.stream_id)
    {
        auto _vstream = _vctx.lock()->getVStreamMap().at((int)args->imms.stream_id); // const qualifier _vctx.lock()
        LOG(INFO) << "get STREAM ID " << (int)args->imms.stream_id;
        checkCudaErrors_lo(cuLaunchKernel(_func, dimGrid.x, dimGrid.y, dimGrid.z, 
            dimBlock.x, dimBlock.y, dimBlock.z,
            0, _vstream->getCUStream(), (void**)kernel_args.data(), 0)); // 0 , stream , args, 0
    }
    else
    {
        LOG(INFO) << "get default STREAM ID " << (int)args->imms.stream_id;
        checkCudaErrors_lo(cuLaunchKernel(_func, dimGrid.x, dimGrid.y, dimGrid.z, 
            dimBlock.x, dimBlock.y, dimBlock.z,
            0, 0, (void**)kernel_args.data(), 0)); // 0 , stream , args, 0
    }

    ch->make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .on_channel()
        .invoke()
        .as_callback();

    auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_start);
    LOG(INFO) << "time for launch kernel server: " << t_usec.count() << std::endl;
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

    ch->revoke(self->_req_call)
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

