#include <cuda_runtime.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <numeric>
#include <pthread.h>

#include "./context.hpp"
// #include "srv_function.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Function;
using namespace fractos;


std::pair<CUresult, std::shared_ptr<impl::Function>>
impl::make_function(std::shared_ptr<test::gpu_Context> ctx_ptr, CUmodule mod, const std::string name)
{
    std::shared_ptr<Function> res;

    auto error = cuCtxSetCurrent(ctx_ptr->_ctx);
    if (error != CUDA_SUCCESS) {
        return std::make_pair(error, res);
    }

    CUfunction func;
    error = cuModuleGetFunction(&func, mod, name.c_str());
    if (error != CUDA_SUCCESS) {
        return std::make_pair(error, res);
    }

    size_t args_total_size = 0;
    std::vector<size_t> args_size;
    for (size_t i = 0; true; i++) {
        size_t offset, size;
        error = cuFuncGetParamInfo(func, i, &offset, &size);
        if (error == CUDA_SUCCESS) {
            args_total_size += size;
            args_size.push_back(size);
        } else if (error == CUDA_ERROR_INVALID_VALUE) {
            error = CUDA_SUCCESS;
            break;
        } else {
            return std::make_pair(error, res);
        }
    }

    res = std::make_shared<Function>(ctx_ptr, func, std::move(args_size), args_total_size);
    res->self = res;
    return std::make_pair(error, res);
}

impl::Function::Function(std::weak_ptr<test::gpu_Context> ctx_ptr, CUfunction func,
                         std::vector<size_t> args_size, size_t args_total_size)
    :func(func)
    ,args_total_size(args_total_size)
    ,args_size(std::move(args_size))
    ,ctx_ptr(ctx_ptr)
{
}

impl::Function::~Function()
{
}

core::future<void>
impl::Function::register_methods(std::shared_ptr<core::channel> ch)
{
    auto self = this->self;

    return ch->make_request_builder<srv_wire_msg::generic::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self, ch](auto& fut) {
            self->req_generic = fut.get();
        });

}

void
impl::Function::handle_generic(auto ch, auto args)
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
    case srv_wire_msg::OP_LAUNCH:
        HANDLE(launch);
        break;
    case srv_wire_msg::OP_DESTROY:
        HANDLE(destroy);
        break;

    default:
        LOG_RES(method)
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
impl::Function::handle_launch(auto ch, auto args)
{
    METHOD(launch);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_CAPS_EXACT(reqb_cont);
    CHECK_IMMS_ALL(reqb_cont);
    CHECK_ARGS_COND(reqb_cont, args->imms_size() == (sizeof(msg::request::imms) + args_total_size));

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    auto ctx_ptr = this->ctx_ptr.lock();
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
        for (size_t i = 0; i < args_size.size(); i++) {
            kernel_args.push_back(args_ptr);
            args_ptr += args_size[i];
        }

        int stream_id = args->imms.stream_id;
        if (stream_id) {
            LOG_FIRST_N(WARNING, 1) << "TODO: add a proper API to query/get stream_ids";
            LOG_FIRST_N(WARNING, 1) << "TODO: return error when stream_id is incorrect";
            auto stream_ptr = ctx_ptr->getVStreamMap().at(stream_id);
            cuerror = cuLaunchKernel(func, dimGrid.x, dimGrid.y, dimGrid.z,
                                     dimBlock.x, dimBlock.y, dimBlock.z,
                                     0, stream_ptr->getCUStream(),
                                     (void**)kernel_args.data(), 0);
        } else {
            cuerror = cuLaunchKernel(func, dimGrid.x, dimGrid.y, dimGrid.z,
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
        .as_callback_log_ignore_continuation_error();
}

void
impl::Function::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    if (not destroy_maybe()) {
        error = wire::ERR_OTHER;
        LOG_RES(method)
            << " error=" << wire::to_string(error)
            << " cuerror=" << cudaGetErrorString((cudaError)cuerror);
        reqb_cont
            .set_imm(&msg::response::imms::error, error)
            .set_imm(&msg::response::imms::cuerror, cuerror)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_continuation_error();
        return;
    }

    ch->revoke(req_generic)
        .then([ch, this, self, args=std::move(args)](auto& fut) {
            auto error = wire::ERR_SUCCESS;
            auto cuerror = CUDA_SUCCESS;
            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << cudaGetErrorString((cudaError)cuerror);
            fut.get();
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();

}

std::string
impl::to_string(const impl::Function& obj)
{
    std::stringstream ss;
    ss << "Function(" << &obj << ")";
    return ss.str();
}
