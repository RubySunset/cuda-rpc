#include <pthread.h>
#include <glog/logging.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

#include "./device.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Device;
using namespace fractos;
using namespace impl;


Device::Device(CUdevice device)
    :device(device)
{
    //fork();
    _destroyed = false;
}

std::shared_ptr<Device> Device::factory(wire::endian::uint8_t value){
    auto res = std::shared_ptr<Device>(new Device(value));
    res->_self = res;
    return res;
}

Device::~Device() {

}

/*
 *  Make handlers for a Device's caps
 */
core::future<void>
Device::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Device;


    auto self = _self;

    return ch->make_request_builder<msg_base::make_context::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            DVLOG(fractos::logging::SERVICE) << "In device register_methods handler";
            self->handle_make_context(std::move(args));
        })
        .on_channel()
        .make_request()
        .then([ch, self](auto& fut) {
            self->req_make_context = fut.get();
            DVLOG(fractos::logging::SERVICE) << "SET req_make_context";
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
            DVLOG(fractos::logging::SERVICE) << "SET req_destroy device";
            self->req_destroy = fut.get();

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
Device::handle_generic(auto ch, auto args)
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
    case srv::wire::Device::OP_GET_ATTRIBUTE:
        HANDLE(get_attribute);
        break;
    case srv::wire::Device::OP_GET_NAME:
        HANDLE(get_name);
        break;
    case srv::wire::Device::OP_GET_UUID:
        HANDLE(get_uuid);
        break;
    case srv::wire::Device::OP_TOTAL_MEM:
        HANDLE(total_mem);
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

template<class T, class U>
constexpr ptrdiff_t
offset_of_member(T U::* member)
{
    return reinterpret_cast<std::ptrdiff_t>(&(((U const volatile*)nullptr)->*member));
}


void
Device::handle_get_attribute(auto ch, auto args)
{
    METHOD(get_attribute);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    int pi;
    auto attrib = (CUdevice_attribute)args->imms.attrib.get();
    auto res = cuDeviceGetAttribute(&pi, attrib, device);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " pi=" << pi;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::pi, pi)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
Device::handle_get_name(auto ch, auto args)
{
    METHOD(get_name);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    size_t name_len = 512;
    char name[name_len];
    auto res = cuDeviceGetName(name, name_len, device);
    name_len = strlen(name);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " name=" << name
        << " len=" << name_len;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::len, name_len)
        .set_imm(&msg::response::imms::name, name, name_len)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
Device::handle_get_uuid(auto ch, auto args)
{
    METHOD(get_uuid);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    static_assert(sizeof(CUuuid) == sizeof(msg::response::imms::uuid));
    CUuuid uuid;
    auto res = cuDeviceGetUuid(&uuid, device);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " uuid=" << srv::wire::to_string(uuid);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(offset_of_member(&msg::response::imms::uuid), (void*)&uuid, sizeof(uuid))
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
Device::handle_total_mem(auto ch, auto args)
{
    METHOD(total_mem);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    size_t bytes;
    auto res = cuDeviceTotalMem(&bytes, device);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " bytes=" << bytes;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::bytes, bytes)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

/*
 *  Destroy a Device, revoke all of its caps
 */
void Device::handle_make_context(auto args) {
    DVLOG(logging::SERVICE) << "CALL handle make_context";
    using msg = ::service::compute::cuda::wire::Device::make_context;
    
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

    unsigned int value = args->imms.flags; // uint32_t

    auto self = _self; // lock()

    VLOG(fractos::logging::SERVICE) << "vctx value is: " << (uint64_t)value;

    auto vctx = std::shared_ptr<test::gpu_Context>(test::gpu_Context::factory(value, device));

    vctx->register_methods(ch)
        .then([this, ch, self, vctx, args=std::move(args), value](auto& fut) {
            fut.get();
            _vctx = vctx;
            // _vdev_map.insert({value, vdev});
            ch->make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS) // test
                .set_cap(&msg::response::caps::generic, vctx->_req_generic)
                .set_cap(&msg::response::caps::make_stream, vctx->_req_stream)
                .set_cap(&msg::response::caps::make_event, vctx->_req_event)
                .set_cap(&msg::response::caps::make_module_data, vctx->_req_module_data) // data
                .set_cap(&msg::response::caps::synchronize, vctx->_req_synchronize)
                .set_cap(&msg::response::caps::destroy, vctx->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}


/*
 *  Destroy a Device, revoke all of its caps
 */
void Device::handle_destroy(auto args) {
    VLOG(fractos::logging::SERVICE) << "CALL handle destroy";
    using msg = ::service::compute::cuda::wire::Device::destroy;

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

    VLOG(fractos::logging::SERVICE) << "Revoke destroy";

    ch->revoke(self->req_make_context)
        .then([ch, self](auto& fut) {
                  fut.get();
                  VLOG(fractos::logging::SERVICE) << "Revoke _req_register_function";
                  return ch->revoke(self->req_destroy);
        })
        .unwrap()
        .then([this, ch, self, args=std::move(args)](auto& fut) {
            fut.get();
            VLOG(fractos::logging::SERVICE) << "Virtual device destroyed";
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
impl::to_string(const Device& obj)
{
    std::stringstream ss;
    ss << "Device(" << &obj << ")";
    return ss.str();
}
