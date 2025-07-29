#include <glog/logging.h>
#include <fractos/common/service/srv_impl.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <string>

#include "./common.hpp"
#include "./service.hpp"
#include "./device.hpp"
#include "./context.hpp"
#include "./library.hpp"
#include "./event.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Service;
using namespace fractos;


impl::Service::Service() {

    // checkCudaErrors(cuInit(0));
}

void impl::Service::request_exit() {
   _requested_exit.store(true); 
}

bool impl::Service::exit_requested() const
{
    return _requested_exit.load();
}

std::shared_ptr<impl::Service> impl::Service::factory() {
    auto res = std::shared_ptr<Service>(new Service());
    res->_self = res;
    return res;
}

/*
 *  The handler for make_device request
 *  Registers all methods that a Device has
 */
core::future<void>
impl::Service::register_service(std::shared_ptr<core::channel> ch)
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

core::future<std::shared_ptr<impl::Device>>
impl::Service::get_or_make_device_ordinal(auto ch, int ordinal)
{
    {
        auto devices_lock = std::shared_lock(_devices_mutex);
        auto it = _ordinal_devices.find(ordinal);
        if (it != _ordinal_devices.end()) {
            return core::make_ready_future(it->second);
        }
    }

    auto self = _self.lock();
    CHECK(self);

    return make_device(ch, self, ordinal)
        .then([this, self](auto& fut) {
            auto [error, cuerror, dev] = fut.get();
            auto cuordinal = dev->get_remote_cuordinal();
            auto cudevice = dev->get_remote_cudevice();

            auto devices_lock = std::unique_lock(_devices_mutex);

            auto res1 = _ordinal_devices.insert(std::make_pair(cuordinal, dev));
            if (not res1.second) {
                auto it = _ordinal_devices.find(cuordinal);
                CHECK(it != _ordinal_devices.end());
                return it->second;
            }

            auto res2 = _devices.insert(std::make_pair(cudevice, dev));
            CHECK(res2.second);

            return dev;
        });
}

std::shared_ptr<impl::Device>
impl::Service::get_device(CUdevice device)
{
    auto devices_lock = std::shared_lock(_devices_mutex);
    auto it = _devices.find(device);
    if (it != _devices.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void
impl::Service::erase_device(std::shared_ptr<Device> dev)
{
    auto devices_lock = std::unique_lock(_devices_mutex);
    {
        auto res = _ordinal_devices.erase(dev->get_remote_cuordinal());
        CHECK(res == 1);
    }
    {
        auto res = _devices.erase(dev->get_remote_cudevice());
        CHECK(res == 1);
    }
}

std::shared_ptr<impl::Context>
impl::Service::get_context(CUcontext cucontext)
{
    auto lock = std::shared_lock(_contexts_mutex);
    auto it = _contexts.find(cucontext);
    if (it != _contexts.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void
impl::Service::insert_context(std::shared_ptr<Context> context)
{
    auto lock = std::unique_lock(_contexts_mutex);
    CHECK(_contexts.insert({context->get_remote_cucontext(), context}).second);
}

void
impl::Service::erase_context(std::shared_ptr<Context> context)
{
    auto lock = std::unique_lock(_contexts_mutex);
    CHECK(_contexts.erase(context->get_remote_cucontext()) == 1);
}

std::shared_ptr<impl::Event>
impl::Service::get_event(CUevent cuevent)
{
    auto lock = std::shared_lock(_events_mutex);
    auto it = _events.find(cuevent);
    if (it != _events.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

void
impl::Service::insert_event(std::shared_ptr<Event> event)
{
    auto lock = std::unique_lock(_events_mutex);
    CHECK(_events.insert({event->get_remote_cuevent(), event}).second);
}

void
impl::Service::erase_event(std::shared_ptr<Event> event)
{
    auto lock = std::unique_lock(_events_mutex);
    CHECK(_events.erase(event->get_remote_cuevent()) == 1);
}


void
impl::Service::handle_connect(auto ch, auto args)
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
            .as_callback_log_ignore_continuation_error();

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
        .as_callback_log_ignore_continuation_error();
}

void
impl::Service::handle_generic(auto ch, auto args)
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

#define HANDLE(name) \
    handle_ ## name(ch, reinterpreted.template operator()<srv_wire_msg:: name ::request>(std::move(args)))

    switch (opcode) {
    CASE_HANDLE(GET_DRIVER_VERSION, get_driver_version);
    CASE_HANDLE(INIT, init);
    CASE_HANDLE(DEVICE_GET, device_get);
    CASE_HANDLE(DEVICE_GET_COUNT, device_get_count);
    CASE_HANDLE(MODULE_GET_LOADING_MODE, module_get_loading_mode);
    CASE_HANDLE(LIBRARY_LOAD_DATA, library_load_data);
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

#undef CASE_HANDLE
}

void
impl::Service::handle_get_driver_version(auto ch, auto args)
{
    METHOD(get_driver_version);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

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
        .as_callback_log_ignore_continuation_error();
}

void
impl::Service::handle_init(auto ch, auto args)
{
    METHOD(init);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

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
        .as_callback_log_ignore_continuation_error();
}


void
impl::Service::handle_device_get(auto ch, auto args)
{
    METHOD(device_get);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    get_or_make_device_ordinal(ch, args->imms.ordinal)
        .then([this, self=_self.lock(), ch, args=std::move(args)](auto& fut) {
            auto dev = fut.get();

            auto error = wire::ERR_SUCCESS;
            CUdevice cudevice;
            auto req = ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error);

            if (dev) {
                cudevice = dev->get_remote_cudevice();
                LOG_RES(method)
                    << " error=" << wire::to_string(error)
                    << " cudevice=" << cudevice
                    << " generic=" << core::to_string(dev->_req_generic);
                req
                    .set_cap(&msg::response::caps::generic, dev->_req_generic);
            } else {
                cudevice = -1;
                LOG_RES(method)
                    << " error=" << wire::to_string(error)
                    << " cudevice=" << cudevice;
                core::cap::request null(core::cap::null_cid);
                req
                    .set_cap(&msg::response::caps::generic, null);
            }

            req
                .set_imm(&msg::response::imms::device, cudevice)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();
}

void
impl::Service::handle_device_get_count(auto ch, auto args)
{
    METHOD(device_get_count);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

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
        .as_callback_log_ignore_continuation_error();
}


void
impl::Service::handle_module_get_loading_mode(auto ch, auto args)
{
    METHOD(module_get_loading_mode);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

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
        .as_callback_log_ignore_continuation_error();
}

void
impl::Service::handle_library_load_data(auto ch, auto args)
{
    METHOD(library_load_data);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_IMMS_ALL(reqb_cont);
    CHECK_CAPS_EXACT(reqb_cont);

    auto self = this->_self.lock();
    CHECK(self);
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    auto send_error = [&]() {
        LOG_RES(method)
            << " error=" << wire::to_string(error)
            << " cuerror=" << get_CUresult_name(cuerror);

        ch->template make_request_builder<msg::response>(args->caps.continuation)
            .set_imm(&msg::response::imms::error, error)
            .set_imm(&msg::response::imms::cuerror, cuerror)
            .on_channel()
            .invoke()
            .as_callback();
    };

    auto contents_size = args->caps.contents.get_size();
    std::shared_ptr<char[]> contents_buffer(new char[contents_size]);
    {
        // passing explicit MR avoids MR creation and prefetching
        auto& mr = ch->get_default_memory_region();
        auto copied_mem = ch->make_memory(contents_buffer.get(), contents_size, mr).get();
        LOG_FIRST_N(WARNING, 1) << "TODO: copy contents asynchronously";
        ch->copy(args->caps.contents, copied_mem).get();
    }


    size_t offset = 0;


    auto jit_options_size = sizeof(CUjit_option) * args->imms.num_jit_options;
    if ((sizeof(args->imms) + offset + jit_options_size) > args->imms_size()) {
        cuerror = CUDA_ERROR_INVALID_VALUE;
        send_error();
        return;
    }
    std::vector<CUjit_option> jit_options(
        (CUjit_option*)&args->imms.data[offset],
        (CUjit_option*)&args->imms.data[offset + jit_options_size]);
    offset += jit_options_size;

    std::vector<void*> jit_values;
    if (args->imms.size_jit_values > 0) {
        LOG(ERROR) << "!!! not implemented";
        cuerror = CUDA_ERROR_UNKNOWN;
        send_error();
        return;
    }


    auto lib_options_size = sizeof(CUlibraryOption) * args->imms.num_lib_options;
    if ((sizeof(args->imms) + offset + lib_options_size) > args->imms_size()) {
        cuerror = CUDA_ERROR_INVALID_VALUE;
        send_error();
        return;
    }
    std::vector<CUlibraryOption> lib_options(
        (CUlibraryOption*)&args->imms.data[offset],
        (CUlibraryOption*)&args->imms.data[offset + lib_options_size]);
    offset += lib_options_size;

    std::vector<void*> lib_values;
    if (args->imms.size_lib_values > 0) {
        LOG(ERROR) << "!!! not implemented";
        cuerror = CUDA_ERROR_UNKNOWN;
        send_error();
        return;
    }


    make_library(ch, contents_buffer,
                 jit_options, jit_values, lib_options, lib_values)
        .then([this, self, ch, args=std::move(args)](auto& fut) {
            auto [error, cuerror, library] = fut.get();

            CUlibrary culibrary = 0;
            if (not error and not cuerror) {
                culibrary = library->get_remote_culibrary();
            }

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " culibrary=" << (void*)culibrary
                << " req_generic=" << core::to_string(library->req_generic);

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::culibrary, (uint64_t)culibrary)
                .set_cap(&msg::response::caps::generic, library->req_generic)
                .on_channel()
                .invoke()
                .as_callback();
        })
        .as_callback();
}


std::string
impl::to_string(const Service& obj)
{
    std::stringstream ss;
    ss << "Service(" << &obj << ")";
    return ss.str();
}
