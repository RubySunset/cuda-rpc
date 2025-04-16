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

    return ch->make_request_builder<msg_base::make_device::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            // Call local service method with the actual handler
            // implementation. We must std::move args since it is a
            // unique_ptr.
            DVLOG(fractos::logging::SERVICE) << "In register_service handler";
            self->handle_make_device(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self, ch](auto& fut) {
                  self->req_make_device = fut.get();
                  DVLOG(fractos::logging::SERVICE) << "SET req_make_device"; // virtual
                  return ch->make_request_builder<msg_base::get_Device::request>(
                      ch->get_default_endpoint(),
                      [self](auto ch, auto args) {
                          self->handle_get_Device(std::move(args));
                      })
                      .on_channel()
                      .make_request();
        })
        .unwrap()
        .then([ch, self](auto& fut) {
                  self->req_get_Device = fut.get();
                  DVLOG(fractos::logging::SERVICE) << "SET req_get_Device";

            return ch->make_request_builder<msg_base::connect::request>(
                ch->get_default_endpoint(),
                [self](auto ch, auto args) {
                    self->handle_connect(ch, std::move(args));
                })
                .on_channel()
                .make_request();
        })
        .unwrap()
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
        << " get_driver_version=" << core::to_string(req_get_driver_version)
        << " make_device=" << core::to_string(req_make_device)
        << " get_device=" << core::to_string(req_get_Device)
        ;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_cap(&msg::response::caps::connect, req_connect)
        .set_cap(&msg::response::caps::generic, req_generic)
        .set_cap(&msg::response::caps::make_device, req_make_device)
        .set_cap(&msg::response::caps::get_device, req_get_Device)
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

template<class T>
struct receive_args_base_type
{
    using type = std::remove_cvref_t<T>::element_type::base_type;
};

#define METHOD(name)                                                    \
    static const std::string method = "handle_" #name;                  \
    using msg = srv::wire::Service:: name;                              \
    {                                                                   \
        using args_type = receive_args_base_type<decltype(args)>::type; \
        static_assert(std::is_same<msg::request, args_type>::value);    \
    }

#define CHECK_ARGS_EXACT()                                              \
    if (not args->has_exactly_args()) {                                 \
        LOG_RES(method) << " error=ERR_OTHER";                          \
        reqb_cont                                                       \
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)      \
            .on_channel()                                               \
            .invoke()                                                   \
            .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring"); \
        return;                                                         \
    }

void
gpu_device_service::handle_get_driver_version(auto ch, auto args)
{
    METHOD(get_driver_version);
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
    METHOD(init);
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
gpu_device_service::handle_module_get_loading_mode(auto ch, auto args)
{
    METHOD(module_get_loading_mode);
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

/*
 *  Actual handler of the make_device request
 *  Initialize a Device and assign caps to it
 *  Return this Device to the frontend service
 */
void gpu_device_service::handle_make_device(auto args) {
    using msg = ::service::compute::cuda::wire::Service::make_device;
    
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

    wire::endian::uint8_t value = args->imms.value;

    auto self = _self.lock();

    VLOG(fractos::logging::SERVICE) << "vdev value is: " << (uint64_t)value;

    auto vdev = std::shared_ptr<gpu_Device>(gpu_Device::factory(value));

    vdev->register_methods(ch)
        .then([this, ch, self, vdev, args=std::move(args), value](auto& fut) {
            fut.get();
            _vdev = vdev;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                .set_cap(&msg::response::caps::make_context, vdev->_req_make_context) // test
                .set_cap(&msg::response::caps::destroy, vdev->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();

}



void gpu_device_service::handle_get_Device(auto args) {
    // namespace msg_base = ::service::compute::cuda::wire::Service;
    using msg = ::service::compute::cuda::wire::Service::get_Device;

    if (not args->has_valid_cap(&msg::request::caps::continuation, core::cap::request_tag)) {
        LOG(ERROR) << "no continuation";
        return;
    }

    std::shared_ptr<core::channel> ch = args->caps_raw[0].get_channel();
    auto vdev = _vdev;//_map[args->imms.value];

    if (args->has_exactly_args() and
        ch->has_object(vdev->_req_destroy).get()) {

        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
            .set_cap(&msg::response::caps::destroy, vdev->_req_destroy)
            .on_channel()
            .invoke()
            .as_callback();

    } else {
        ch->make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback();
    }
}

std::string
test::to_string(const gpu_device_service& obj)
{
    std::stringstream ss;
    ss << "Service(" << &obj << ")";
    return ss.str();
}
