// #include "device_service.hpp"
#include <glog/logging.h>
#include <srv_service.hpp>
// #include <srv_device.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <string>

#include "./common.hpp"

using namespace fractos;
using namespace ::test;
namespace srv = fractos::service::compute::cuda;


gpu_device_service::gpu_device_service() {

    // checkCudaErrors(cuInit(0));
}

void gpu_device_service::request_exit() {
   _requested_exit.store(true); 
}

bool gpu_device_service::exit_requested() const
{
    return _requested_exit.load();
}

std::shared_ptr<gpu_device_service> gpu_device_service::factory() {
    auto res = std::shared_ptr<gpu_device_service>(new gpu_device_service());
    res->_self = res;
    return res;
}

gpu_device_service::~gpu_device_service() {


}
/*
 *  The handler for make_device request
 *  Registers all methods that a Device has
 */
core::future<void>
gpu_device_service::register_service(std::shared_ptr<core::channel> ch)
{
    // namespace msg_base = ::service::compute::detail::device_service;
    namespace msg_base = ::service::compute::cuda::wire::Service;

    auto self = _self.lock();

    return ch->make_request_builder<msg_base::connect::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_connect(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->req_connect = fut.get();

            return ch->make_request_builder<msg_base::generic::request>(
                ch->get_default_endpoint(),
                [self](auto ch, auto args) {
                    self->handle_generic(ch, std::move(args));
                })
                .on_channel()
                .make_request();
        })
        .unwrap()
        .then([self](auto& fut) {
            self->req_generic = fut.get();
        });
}

void
gpu_device_service::handle_connect(auto ch, auto args)
{
    static const std::string method = "handle_connect";
    using msg = ::service::compute::cuda::wire::Service::connect;

    LOG_REQ(method)
        << srv::wire::to_string(*args);

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG_RES(method)
            << " [error] request without continuation, ignoring";
        return;
    }

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);

    if (not args->has_exactly_args()) {
        LOG_RES(method)
            << " error=ERR_OTHER";

        reqb_cont
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");

        return;
    }

    auto error = wire::ERR_SUCCESS;

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " connect=" << core::to_string(req_connect)
        << " generic=" << core::to_string(req_generic)
        ;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_cap(&msg::response::caps::connect, req_connect)
        .set_cap(&msg::response::caps::generic, req_generic)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}

void
gpu_device_service::handle_generic(auto ch, auto args)
{
    static const std::string method = "handle_generic";
    using msg = ::service::compute::cuda::wire::Service::generic;

    auto opcode = srv::wire::Service::OP_INVALID;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG_OP(method)
            << " [error] request without continuation, ignoring";
        return;
    } else if (args->has_imm(&msg::request::imms::opcode)) {
        opcode = static_cast<srv::wire::Service::generic_opcode>(args->imms.opcode.get());
    }

    auto reinterpreted = []<class T>(auto args) {
        using ptr = core::receive_args<T>;
        return std::unique_ptr<ptr>(reinterpret_cast<ptr*>(args.release()));
    };

#define HANDLE(name) \
    handle_ ## name(ch, reinterpreted.template operator()<srv::wire::Service:: name ::request>(std::move(args)))

    switch (opcode) {
    case srv::wire::Service::OP_GET_DRIVER_VERSION:
        HANDLE(get_driver_version);
        break;

    case srv::wire::Service::OP_INIT:
        HANDLE(init);
        break;

    case srv::wire::Service::OP_DEVICE_GET:
        HANDLE(device_get);
        break;
    case srv::wire::Service::OP_DEVICE_GET_COUNT:
        HANDLE(device_get_count);
        break;

    case srv::wire::Service::OP_MODULE_GET_LOADING_MODE:
        HANDLE(module_get_loading_mode);
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
gpu_device_service::handle_get_driver_version(auto ch, auto args)
{
    METHOD(Service, get_driver_version);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    int version;
    auto res = cuDriverGetVersion(&version);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " value=" << version;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::value, version)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}

void
gpu_device_service::handle_init(auto ch, auto args)
{
    METHOD(Service, init);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    auto res = cuInit(args->imms.flags);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}


void
gpu_device_service::handle_device_get(auto ch, auto args)
{
    METHOD(Service, device_get);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    LOG(WARNING) << "TODO: move cuDeviceGet here";

    auto self = _self.lock();

    auto return_device = [this, self, ch](auto args, auto dev) {
        auto error = wire::ERR_SUCCESS;
        LOG_RES(method)
            << " error=" << wire::to_string(error)
            << " device=" << dev->device
            << " generic=" << core::to_string(dev->req_generic)
            << " make_context=" << core::to_string(dev->req_make_context)
            << " destroy=" << core::to_string(dev->req_destroy);

        ch->template make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, error)
            .set_imm(&msg::response::imms::device, dev->device)
            .set_cap(&msg::response::caps::generic, dev->req_generic)
            .set_cap(&msg::response::caps::make_context, dev->req_make_context)
            .set_cap(&msg::response::caps::destroy, dev->req_destroy)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
    };

    if (not _vdev) {
        auto vdev = gpu_Device::factory(args->imms.ordinal);
        vdev->register_methods(ch)
            .then([this, self, return_device, vdev, args=std::move(args)](auto& fut) mutable {
                fut.get();

                LOG(WARNING) << "TODO: track all devices atomically";
                self->_vdev = vdev;

                return_device(std::move(args), _vdev);
            })
            .as_callback();
    } else {
        return_device(std::move(args), _vdev);
    }
}

void
gpu_device_service::handle_device_get_count(auto ch, auto args)
{
    METHOD(Service, device_get_count);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    int count;
    auto res = cuDeviceGetCount(&count);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " count=" << std::to_string(count);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::count, count)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}


void
gpu_device_service::handle_module_get_loading_mode(auto ch, auto args)
{
    METHOD(Service, module_get_loading_mode);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT();

    CUmoduleLoadingMode mode;
    auto res = cuModuleGetLoadingMode(&mode);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " mode=" << std::to_string(mode);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::mode, mode)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring");
}


std::string
test::to_string(const gpu_device_service& obj)
{
    std::stringstream ss;
    ss << "Service(" << &obj << ")";
    return ss.str();
}
