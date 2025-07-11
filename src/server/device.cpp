#include <cuda_runtime.h>
#include <pthread.h>
#include <glog/logging.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

#include "./device.hpp"
#include "./context.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Device;
using namespace fractos;


std::pair<CUresult, std::shared_ptr<impl::Device>>
impl::make_device(int ordinal)
{
    std::shared_ptr<Device> res;

    CUdevice device;
    auto cuerr = cuDeviceGet(&device, ordinal);
    if (cuerr != CUDA_SUCCESS) {
        return std::make_pair(cuerr, res);
    }

    res = std::make_shared<Device>(device);
    res->self = res;
    return std::make_pair(cuerr, res);
}

impl::Device::Device(CUdevice device)
    :device(device)
{
}

impl::Device::~Device()
{
}

core::future<void>
impl::Device::register_methods(std::shared_ptr<core::channel> ch)
{
    namespace msg_base = ::service::compute::cuda::wire::Device;


    auto self = this->self.lock();

    return ch->make_request_builder<msg_base::generic::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self](auto& fut) {
            self->req_generic = fut.get();
        });
}

void
impl::Device::handle_generic(auto ch, auto args)
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
    CASE_HANDLE(GET_ATTRIBUTE, get_attribute);
    CASE_HANDLE(GET_NAME, get_name);
    CASE_HANDLE(GET_UUID, get_uuid);
    CASE_HANDLE(TOTAL_MEM, total_mem);
    CASE_HANDLE(GET_PROPERTIES, get_properties);
    CASE_HANDLE(CTX_CREATE, ctx_create);
    CASE_HANDLE(DESTROY, destroy);
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
impl::Device::handle_get_attribute(auto ch, auto args)
{
    METHOD(get_attribute);
    LOG_REQ(method) << srv_wire::to_string(*args);

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
impl::Device::handle_get_name(auto ch, auto args)
{
    METHOD(get_name);
    LOG_REQ(method) << srv_wire::to_string(*args);

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
impl::Device::handle_get_uuid(auto ch, auto args)
{
    METHOD(get_uuid);
    LOG_REQ(method) << srv_wire::to_string(*args);

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
        << " uuid=" << srv_wire::to_string(uuid);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(offset_of_member(&msg::response::imms::uuid), (void*)&uuid, sizeof(uuid))
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Device::handle_total_mem(auto ch, auto args)
{
    METHOD(total_mem);
    LOG_REQ(method) << srv_wire::to_string(*args);

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

void
impl::Device::handle_get_properties(auto ch, auto args)
{
    // this is not a cuda driver operation, but it's easier and faster than
    // a sequence of calls to cuDeviceGetAttribute()

    METHOD(get_properties);
    LOG_REQ(method) << srv_wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    cudaDeviceProp prop;
    cuerror = (CUresult)cudaGetDeviceProperties(&prop, device);

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror)
        << " data_size=" << sizeof(prop);

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .set_imm(&msg::response::imms::data_size, sizeof(prop))
        .set_imm(&msg::response::imms::data, &prop, sizeof(prop))
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Device::handle_ctx_create(auto ch, auto args)
{
    METHOD(ctx_create);
    LOG_REQ(method) << srv_wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self.lock();
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    unsigned int flags = args->imms.flags;

    auto ctx_ptr = impl::Context::factory(flags, device);

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << cudaGetErrorString((cudaError)cuerror)
        << " ctx_ptr=" << to_string(*ctx_ptr);

    ctx_ptr->register_methods(ch)
        .then([this, ch, args=std::move(args), self, ctx_ptr, error, cuerror](auto& fut) {
            fut.get();
            self->ctx_ptr = ctx_ptr;
            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_cap(&msg::response::caps::generic, ctx_ptr->_req_generic)
                .set_cap(&msg::response::caps::synchronize, ctx_ptr->_req_synchronize)
                .set_cap(&msg::response::caps::destroy, ctx_ptr->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
              })
        .as_callback();
}

void
impl::Device::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv_wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self.lock();
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

    ch->revoke(self->req_generic)
        .then([this, ch, args=std::move(args), self, error, cuerror](auto& fut) {
            fut.get();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << cudaGetErrorString((cudaError)cuerror);

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
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
