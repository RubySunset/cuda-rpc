#include <cuda_runtime.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./service.hpp"
#include "./device.hpp"
#include "./context.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Device;
using namespace fractos;


std::string
impl::to_string(const Device& obj)
{
    std::stringstream ss;
    ss << "Device(" << obj.get_remote_cudevice() << ")";
    return ss.str();
}


core::future<std::tuple<wire::error_type, CUresult, std::shared_ptr<impl::Device>>>
impl::make_device(std::shared_ptr<fractos::core::channel> ch,
                  std::shared_ptr<Service> service,
                  int cuordinal)
{
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    std::shared_ptr<Device> res;

    CUdevice cudevice;
    auto cuerr = cuDeviceGet(&cudevice, cuordinal);
    if (cuerr != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    res = std::make_shared<Device>();
    res->_remote_cuordinal = cuordinal;
    res->cudevice = cudevice;
    res->service = service;
    res->self = res;

    return ch->make_request_builder<srv_wire_msg::generic::request>(
        ch->get_default_endpoint(),
        [res](auto ch, auto args) {
            res->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([res](auto& fut) {
            res->_req_generic = fut.get();
        })
        .then([error, cuerror, res](auto& fut) mutable {
            try {
                fut.get();
            } catch (const core::generic_error& e) {
                error = (wire::error_type)e.error;
            }

            if (error or cuerror) {
                LOG(FATAL) << "TODO: undo Device and return error";
            }

            return std::make_tuple(error, cuerror, res);
        });
}


int
impl::Device::get_remote_cuordinal() const
{
    return _remote_cuordinal;
}

CUdevice
impl::Device::get_remote_cudevice() const
{
    return (CUdevice)get_remote_cuordinal();
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
    auto res = cuDeviceGetAttribute(&pi, attrib, cudevice);

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
    auto res = cuDeviceGetName(name, name_len, cudevice);
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
    auto res = cuDeviceGetUuid(&uuid, cudevice);

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
    auto res = cuDeviceTotalMem(&bytes, cudevice);

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
    cuerror = (CUresult)cudaGetDeviceProperties(&prop, cudevice);

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

    auto self = this->self;
    unsigned int flags = args->imms.flags;

    make_context(ch, self, flags)
        .then([this, self, ch, args=std::move(args)](auto& fut) {
            auto [error, cuerror, ctx] = fut.get();

            CUcontext cucontext = 0;
            if (not error and not cuerror) {
                cucontext = ctx->get_remote_cucontext();
            }

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " cucontext=" << (void*)cucontext
                << " req_generic=" << core::to_string(ctx->_req_generic);

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::cucontext, (uint64_t)cucontext)
                .set_cap(&msg::response::caps::generic, ctx->_req_generic)
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

    service->erase_device(self);

    ch->revoke(self->_req_generic)
        .then([this, self, ch, args=std::move(args), error, cuerror](auto& fut) {
            fut.get();

            self->self.reset();

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
