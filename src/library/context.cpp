#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <fstream>
#include <utility>

#include <./common.hpp>
#include <context_impl.hpp>
#include <module_impl.hpp>
#include <memory_impl.hpp>
#include <stream_impl.hpp>
#include <event_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Context;
using namespace fractos;


#define IMPL_CLASS impl::Context
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Context>;


std::shared_ptr<clt::Context>
impl::make_context(std::shared_ptr<fractos::core::channel> ch,
                   std::shared_ptr<clt::Device> device,
                   fractos::core::cap::request req_generic,
                   fractos::core::cap::request req_module_data,
                   fractos::core::cap::request req_ctx_sync,
                   fractos::core::cap::request req_ctx_destroy)
{
    auto state = std::make_shared<impl::ContextState>();
    state->device = device;
    state->req_generic = std::move(req_generic);
    state->req_module_data = std::move(req_module_data);
    state->req_ctx_sync = std::move(req_ctx_sync);
    state->req_ctx_destroy = std::move(req_ctx_destroy);
    state->context_ptr = std::unique_ptr<char[]>(new char[512]);
    state->context = (CUcontext)state->context_ptr.get();

    return impl::Context::make(ch, state);
}

core::future<void>
impl::ContextState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_ctx_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response_ptr(self)
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

CUcontext
clt::Context::get_context() const
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->context;
}

core::future<unsigned int>
clt::Context::get_api_version()
{
    METHOD(get_api_version);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_API_VERSION)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return args->imms.version.get();
        });
}

std::shared_ptr<clt::Device>
clt::Context::get_device()
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->device.lock();
}

core::future<size_t>
clt::Context::get_limit(CUlimit limit)
{
    METHOD(get_limit);
    LOG_REQ(method)
        << " limit=" << limit;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_LIMIT)
        .set_imm(&msg::request::imms::limit, limit)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return args->imms.value.get();
        });
}

core::future<std::shared_ptr<clt::Memory>>
clt::Context::mem_alloc(size_t size)
{
    METHOD(mem_alloc);
    LOG_REQ(method)
        << " size=" << size;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MEM_ALLOC)
        .set_imm(&msg::request::imms::size, size)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self, size](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            CUdeviceptr address = (CUdeviceptr)args->imms.address.get();

            return impl::make_memory(
                ch,
                address,
                size,
                std::move(args->caps.destroy),
                std::move(args->caps.memory));
        });
}


core::future<std::shared_ptr<clt::Memory>>
clt::Context::make_memory(uint64_t size)
{
    return mem_alloc(size);
}


core::future<std::shared_ptr<clt::Stream>>
clt::Context::stream_create(CUstream_flags flags)
{
    METHOD(stream_create);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_STREAM_CREATE)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_stream(
                ch,
                (CUstream)args->imms.custream.get(),
                std::move(args->caps.generic));
        });
}


core::future<std::shared_ptr<clt::Event>>
clt::Context::make_event(fractos::wire::endian::uint32_t flags)
{
    return event_create((CUevent_flags)flags.get());
}

core::future<std::shared_ptr<clt::Event>>
clt::Context::event_create(CUevent_flags flags)
{
    METHOD(event_create);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_EVENT_CREATE)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self, flags](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_event(
                ch,
                std::move(args->caps.destroy));
        });
}

core::future<std::shared_ptr<clt::Module>>
clt::Context::module_load(const std::string path)
{
    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    std::ifstream file(path, std::ifstream::ate | std::ifstream::binary);
    if (not file.is_open()) {
        return core::make_exceptional_future<std::shared_ptr<clt::Module>>(
            CudaError(CUDA_ERROR_FILE_NOT_FOUND));
    }
    auto size = file.tellg();
    file.seekg(0);

    std::shared_ptr<char[]> contents(new char[size]);
    CHECK(file.rdbuf()->sgetn(contents.get(), size) == size);

    // RO cap, synchronously prefetch default MR to avoid unexpected perm errors
    auto& mr = pimpl.ch->get_default_memory_region();
    mr.prefetch(fractos::core::memory_region::prefetch_type::ODP_RD_SYNC,
                (const void*)contents.get(), size);
    return pimpl.ch->make_memory((const void*)contents.get(), size, mr)
        .then([this, self, contents](auto& fut) {
            auto mem = fut.get();
            return this->make_module_data(mem, 0)
                .then([contents, mem=std::move(mem)](auto& fut) mutable {
                    return fut.get();
                });
        })
        .unwrap();
}

core::future<std::shared_ptr<clt::Module>>
clt::Context::make_module_data(core::cap::memory& contents, uint64_t module_id)
{
    METHOD(make_module_data);
    LOG_REQ(method)
        << " contents=" << core::to_string(contents)
        << " module_id=" << module_id;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_module_data)
         .set_imm(&msg::request::imms::module_id, module_id)
        // .set_imm(offsetof(msg::request::imms, file_name), file_name.c_str(), file_name.size())
        .set_cap(&msg::request::caps::continuation, resp)
        .set_cap(&msg::request::caps::cuda_file, contents)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([self, module_id](auto& fut) { // function_name
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_module(
                ch,
                std::move(args->caps.generic),
                std::move(args->caps.get_function),
                std::move(args->caps.destroy));
        });
}

core::future<void>
clt::Context::synchronize()
{
    METHOD(synchronize);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_ctx_sync)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


std::string
clt::to_string(const clt::Context& obj)
{
    auto& pimpl = impl::Context::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Context& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::ContextState& obj)
{
    std::stringstream ss;
    ss << "cuda::Context(" << &obj << ")";
    return ss.str();
}
