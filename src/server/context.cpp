#include <fractos/common/service/srv_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <fstream>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./device.hpp"
#include "./context.hpp"
#include "./stream.hpp"
#include "./event.hpp"
#include "./module.hpp"
#include "./memory.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Context;
using namespace fractos;


std::string
impl::to_string(const Context& obj)
{
    std::stringstream ss;
    ss << "Context(" << obj.get_remote_cucontext() << ")";
    return ss.str();
}


core::future<std::tuple<wire::error_type, CUresult, std::shared_ptr<impl::Context>>>
impl::make_context(std::shared_ptr<fractos::core::channel> ch,
                   std::shared_ptr<Device> device,
                   unsigned int flags)
{
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    std::shared_ptr<Context> res;

    CUcontext cucontext;
    cuerror = cuCtxCreate(&cucontext, flags, device->cudevice);
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    res = std::make_shared<Context>();
    res->cucontext = cucontext;
    res->device = device;
    res->self = res;

    return ch->make_request_builder<srv_wire_msg::generic::request>(
        ch->get_default_endpoint(),
        [self=res](auto ch, auto args) {
            self->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([self=res](auto& fut) {
            self->_req_generic = fut.get();
        })
        .then([error, cuerror, res](auto& fut) mutable {
            try {
                fut.get();
            } catch (const core::generic_error& e) {
                error = (wire::error_type)e.error;
            }

            if (error or cuerror) {
                LOG(FATAL) << "TODO: undo Context and return error";
            }

            return std::make_tuple(error, cuerror, res);
        });
}

CUcontext
impl::Context::get_remote_cucontext() const
{
    return (CUcontext)this;
}

std::shared_ptr<impl::Stream>
impl::Context::get_stream(CUstream stream)
{
    auto lock = std::unique_lock(_stream_map_mutex);
    auto it = _stream_map.find(stream);
    if (it == _stream_map.end()) {
        return nullptr;
    } else {
        return it->second;
    }
}

void
impl::Context::insert_stream(std::shared_ptr<Stream> stream)
{
    auto lock = std::unique_lock(_stream_map_mutex);
    CHECK(_stream_map.insert({stream->custream, stream}).second);
}

void
impl::Context::erase_stream(std::shared_ptr<Stream> stream)
{
    auto lock = std::unique_lock(_stream_map_mutex);
    CHECK(_stream_map.erase(stream->custream) == 1);
}

void
impl::Context::insert_event(std::shared_ptr<Event> event)
{
    auto lock = std::unique_lock(_event_map_mutex);
    CHECK(_event_map.insert({event->cuevent, event}).second);
}

void
impl::Context::erase_event(std::shared_ptr<Event> event)
{
    auto lock = std::unique_lock(_event_map_mutex);
    CHECK(_event_map.erase(event->cuevent) == 1);
}


void
impl::Context::handle_generic(auto ch, auto args)
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
    CASE_HANDLE(GET_API_VERSION, get_api_version);
    CASE_HANDLE(GET_LIMIT, get_limit);
    CASE_HANDLE(MODULE_LOAD_DATA, module_load_data);
    CASE_HANDLE(MEM_ALLOC, mem_alloc);
    CASE_HANDLE(MEM_GET_INFO, mem_get_info);
    CASE_HANDLE(MEMSET, memset);
    CASE_HANDLE(STREAM_CREATE, stream_create);
    CASE_HANDLE(EVENT_CREATE, event_create);
    CASE_HANDLE(SYNCHRONIZE, synchronize);
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

void
impl::Context::handle_get_api_version(auto ch, auto args)
{
    METHOD(get_api_version);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    unsigned int version;
    auto res = cuCtxGetApiVersion(cucontext, &version);

    auto error = wire::ERR_SUCCESS;
    if (res != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " version=" << version;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::version, version)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Context::handle_get_limit(auto ch, auto args)
{
    METHOD(get_limit);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    CUlimit limit = (CUlimit)args->imms.limit.get();

    auto error = wire::ERR_SUCCESS;
    auto cuerr = CUDA_SUCCESS;
    size_t value = 0;

    cuerr = cuCtxSetCurrent(cucontext);
    if (cuerr != CUDA_SUCCESS) {
        goto out;
    }

    cuerr = cuCtxGetLimit(&value, limit);

out:
    if (cuerr != CUDA_SUCCESS) {
        error = wire::ERR_OTHER;
    }

    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " value=" << value;

    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::value, value)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Context::handle_mem_alloc(auto ch, auto args)
{
    METHOD(mem_alloc);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    size_t size = args->imms.size.get();

    make_memory(ch, self, size)
        .then([this, self, ch, args=std::move(args)](auto& fut) {
            auto [error, cuerror, res] = fut.get();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " address=" << (void*)res->cuptr
                << " memory=" << core::to_string(res->memory)
                << " req_generic=" << core::to_string(res->req_generic);

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::address, res->cuptr)
                .set_cap(&msg::response::caps::memory, res->memory)
                .set_cap(&msg::response::caps::generic, res->req_generic)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();
}

void
impl::Context::handle_mem_get_info(auto ch, auto args)
{
    METHOD(mem_get_info);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    size_t free = 0, total = 0;

    cuerror = cuCtxSetCurrent(cucontext);
    if (cuerror != CUDA_SUCCESS) {
        goto out;
    }

    cuerror = cuMemGetInfo(&free, &total);

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror)
        << " free=" << free
        << " total=" << total;

    ch->template make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .set_imm(&msg::response::imms::free, free)
        .set_imm(&msg::response::imms::total, total)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

static inline
bool
is_1d(auto& args)
{
    if (args->imms.row_count == 0) {
        return true;
    } else {
        return false;
    }
}

void
impl::Context::handle_memset(auto ch, auto args)
{
    METHOD(memset);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    auto addr = (CUdeviceptr)args->imms.addr;
    auto custream_arg = (CUstream)args->imms.custream.get();

    std::shared_ptr<Stream> stream_ptr;
    CUstream custream;

    if (custream_arg == 0) {
        custream = custream_arg;
        cuerror = cuCtxSetCurrent(cucontext);
        if (cuerror != CUDA_SUCCESS) {
            goto out;
        }
    } else {
        stream_ptr = get_stream(custream_arg);
        if (not stream_ptr) {
            cuerror = CUDA_ERROR_INVALID_HANDLE;
            goto out;
        }
        custream = stream_ptr->custream;
    }

    switch ((unsigned int)args->imms.value_bytes) {
    case 1: // 8B
        if (is_1d(args)) {
            cuerror = cuMemsetD8Async(addr,
                                      args->imms.value.get() & 0xff,
                                      args->imms.row_elems.get(),
                                      custream);
        } else {
            cuerror = cuMemsetD2D8Async(addr,
                                        args->imms.row_pad.get(),
                                        args->imms.value.get() & 0xff,
                                        args->imms.row_elems.get(),
                                        args->imms.row_count.get(),
                                        custream);
        }
        break;
    case 2: // 16B
        if (is_1d(args)) {
            cuerror = cuMemsetD16Async(addr,
                                       args->imms.value.get() & 0xffff,
                                       args->imms.row_elems.get(),
                                       custream);
        } else {
            cuerror = cuMemsetD2D16Async(addr,
                                         args->imms.row_pad.get(),
                                         args->imms.value.get() & 0xffff,
                                         args->imms.row_elems.get(),
                                         args->imms.row_count.get(),
                                         custream);
        }
        break;
    case 4: // 32B
        if (is_1d(args)) {
            cuerror = cuMemsetD32Async(addr,
                                       args->imms.value.get() & 0xffffffff,
                                       args->imms.row_elems.get(),
                                       custream);
        } else {
            cuerror = cuMemsetD2D32Async(addr,
                                         args->imms.row_pad.get(),
                                         args->imms.value.get() & 0xffffffff,
                                         args->imms.row_elems.get(),
                                         args->imms.row_count.get(),
                                         custream);
        }
        break;
    }

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror);

    ch->template make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Context::handle_stream_create(auto ch, auto args)
{
    METHOD(stream_create);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    CUstream_flags flags = (CUstream_flags)args->imms.flags.get();

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    auto [cuerr, stream] = impl::make_stream(self, flags);
    if (cuerr != CUDA_SUCCESS) {
        goto out_err;
    }

    insert_stream(stream);

    stream->register_methods(ch)
        .then([this, self, ch, args=std::move(args), error, cuerror, stream](auto& fut) {
            fut.get();

            auto custream = stream->get_remote_custream();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " custream=" << (void*)custream;

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::custream, (uint64_t)custream)
                .set_cap(&msg::response::caps::generic, stream->req_generic)
                .on_channel()
                .invoke()
                .as_callback_log_ignore_continuation_error();
        })
        .as_callback();

    return;

out_err:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror)
        << " custream=" << (void*)0;

    ch->template make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}


void
impl::Context::handle_event_create(auto ch, auto args)
{
    METHOD(event_create);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    CUevent_flags flags = (CUevent_flags)args->imms.flags.get();

    make_event(ch, self, flags)
        .then([this, self, ch, args=std::move(args)](auto& fut) {
            auto [error, cuerror, res] = fut.get();

            CUevent cuevent = 0;
            if (not error and not cuerror) {
                cuevent = res->get_remote_cuevent();
            }

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " cuevent=" << (void*)cuevent
                << " req_generic=" << core::to_string(res->req_generic);

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::cuevent, (uint64_t)cuevent)
                .set_cap(&msg::response::caps::generic, res->req_generic)
                .on_channel()
                .invoke()
                .as_callback();
        })
        .as_callback();
}


void
impl::Context::handle_module_load_data(auto ch, auto args)
{
    METHOD(module_load_data);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    auto contents_size = args->caps.contents.get_size();
    std::shared_ptr<char[]> contents_buffer(new char[contents_size]);
    {
        // passing explicit MR avoids MR creation and prefetching
        auto& mr = ch->get_default_memory_region();
        auto copied_mem = ch->make_memory(contents_buffer.get(), contents_size, mr).get();
        ch->copy(args->caps.contents, copied_mem).get();
    }

    auto mod = std::shared_ptr<Module>(Module::factory(cucontext, contents_buffer, contents_size, self));

    mod->register_methods(ch)
        .then([this, self, ch, args=std::move(args), error, cuerror, mod](auto& fut) {
            fut.get();

            LOG_RES(method)
                << " error=" << wire::to_string(error)
                << " cuerror=" << get_CUresult_name(cuerror)
                << " cumodule=" << (void*)mod->get_remote_cumodule();

            ch->template make_request_builder<msg::response>(args->caps.continuation)
                .set_imm(&msg::response::imms::error, error)
                .set_imm(&msg::response::imms::cuerror, cuerror)
                .set_imm(&msg::response::imms::cumodule, (uint64_t)mod->get_remote_cumodule())
                .set_cap(&msg::response::caps::generic, mod->_req_generic)
                .set_cap(&msg::response::caps::get_function, mod->_req_get_func)
                .set_cap(&msg::response::caps::destroy, mod->_req_destroy)
                .on_channel()
                .invoke()
                .as_callback();
        })
        .as_callback();
}

void
impl::Context::handle_synchronize(auto ch, auto args)
{
    METHOD(synchronize);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    cuerror = cuCtxSetCurrent(cucontext);
    if (cuerror != CUDA_SUCCESS) {
        goto out;
    }

    cuerror = cuCtxSynchronize();

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror);

    ch->template make_request_builder<msg::response>(args->caps.continuation)
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}

void
impl::Context::handle_destroy(auto ch, auto args)
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
        goto out;
    }

    {
        auto move_into_new = [this](auto& map_from, auto& map_mutex) {
            std::decay_t<decltype(map_from)> map_to;
            {
                auto lock = std::unique_lock(map_mutex);
                map_to.merge(map_from);
            }
            return map_to;
        };

        auto call_destroy_maybe = [this, ch](auto& futures_to, auto& map_from) {
            futures_to.reserve(futures_to.size() + map_from.size());
            std::transform(map_from.begin(), map_from.end(),
                           std::inserter(futures_to, std::end(futures_to)),
                           [ch](auto& elem) {
                               return elem.second->destroy_maybe(ch);
                           });
        };

        std::vector<core::future<std::tuple<wire::error_type, CUresult>>> destroy_futures;

        auto streams = move_into_new(_stream_map, _stream_map_mutex);
        call_destroy_maybe(destroy_futures, streams);

        auto events = move_into_new(_event_map, _event_map_mutex);
        call_destroy_maybe(destroy_futures, events);

        core::when_all(std::move(destroy_futures))
            .then([ch, this, self, args=std::move(args)](auto& fut) mutable {
                auto destroyed = fut.get();
                for (auto& elem : destroyed) {
                    auto [error, cuerror] = elem.get();
                    CHECK(not error);
                    CHECK(not cuerror);
                }

                ch->revoke(_req_generic)
                    .then([ch, this, self, args=std::move(args)](auto& fut) {
                        fut.get();

                        auto error = wire::ERR_SUCCESS;
                        auto cuerror = cuCtxDestroy(cucontext);

                        self->self.reset();

                        LOG_RES(method)
                            << " error=" << wire::to_string(error)
                            << " cuerror=" << get_CUresult_name(cuerror);

                        ch->template make_request_builder<msg::response>(args->caps.continuation)
                            .set_imm(&msg::response::imms::error, error)
                            .set_imm(&msg::response::imms::cuerror, cuerror)
                            .on_channel()
                            .invoke()
                            .as_callback_log_ignore_continuation_error();
                    })
                    .as_callback();
            })
            .as_callback();
    }
    return;

out:
    LOG_RES(method)
        << " error=" << wire::to_string(error)
        << " cuerror=" << get_CUresult_name(cuerror);
    reqb_cont
        .set_imm(&msg::response::imms::error, error)
        .set_imm(&msg::response::imms::cuerror, cuerror)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_continuation_error();
}
