#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <fstream>
#include <cstdint>
#include <utility>

#include <./common.hpp>
#include <context_impl.hpp>
#include "./library_impl.hpp"
#include "fractos/core/types.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Context;
using namespace fractos;


#define IMPL_CLASS impl::Context
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Context>;

// Declare that other CltBase specializations are instantiated elsewhere
extern template class fractos::common::service::CltBase<clt::Stream>;
extern template class fractos::common::service::CltBase<clt::Event>;

// Include other impl headers after IMPL_CLASS section to avoid template instantiation conflicts
#include <module_impl.hpp>
#include <memory_impl.hpp>
#include <stream_impl.hpp>
#include <event_impl.hpp>
#include <cublas_impl.hpp>


std::shared_ptr<clt::Context>
impl::make_context(std::shared_ptr<fractos::core::channel> ch,
                   std::shared_ptr<clt::Device> device,
                   CUcontext cucontext,
                   fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::ContextState>();
    state->device = device;
    state->req_generic = std::move(req_generic);
    state->context_ptr = std::unique_ptr<char[]>(new char[512]);
    state->cucontext = cucontext;

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
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

CUcontext
clt::Context::get_context() const
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->cucontext;
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
                std::move(args->caps.memory),
                std::move(args->caps.generic));
        });
}

core::future<std::pair<size_t, size_t>>
clt::Context::mem_get_info() const
{
    METHOD(mem_get_info);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MEM_GET_INFO)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            size_t free = args->imms.free.get();
            size_t total = args->imms.total.get();

            return std::make_pair(free, total);
        });
}

auto
impl::ContextState::memset(std::shared_ptr<core::channel> ch,
                           CUdeviceptr addr,
                           uint64_t row_elems, uint64_t row_pad, uint64_t row_count,
                           uint64_t value, uint8_t value_bytes,
                           clt::Stream& stream)
{
    CUstream custream = stream.get_stream();

    METHOD(memset);
    LOG_REQ(method)
        << " addr=" << (void*)addr
        << " row_elems=" << row_elems
        << " row_pad=" << row_pad
        << " row_count=" << row_count
        << " value=" << value
        << " value_bytes=" << value_bytes
        << " custream=" << (void*)custream;

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(this->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MEMSET)
        .set_imm(&msg::request::imms::addr, addr)
        .set_imm(&msg::request::imms::row_elems, row_elems)
        .set_imm(&msg::request::imms::row_pad, row_pad)
        .set_imm(&msg::request::imms::row_count, row_count)
        .set_imm(&msg::request::imms::value, value)
        .set_imm(&msg::request::imms::value_bytes, value_bytes)
        .set_imm(&msg::request::imms::custream, (uint64_t)custream)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, uint8_t val, size_t width, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, 0, 0, val, 1, stream);
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, uint16_t val, size_t width, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, 0, 0, val, 2, stream);
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, uint32_t val, size_t width, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, 0, 0, val, 4, stream);
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, size_t pitch, uint8_t val, size_t width, size_t height, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, pitch, height, val, 1, stream);
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, size_t pitch, uint16_t val, size_t width, size_t height, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, pitch, height, val, 2, stream);
}

core::future<void>
clt::Context::memset(CUdeviceptr addr, size_t pitch, uint32_t val, size_t width, size_t height, Stream& stream)
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->memset(pimpl.ch, addr, width, pitch, height, val, 4, stream);
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
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_stream(
                *this,
                ch,
                (CUstream)args->imms.custream.get(),
                std::move(args->caps.generic));
        });
}

std::shared_ptr<clt::Stream>
clt::Context::get_legacy_default_stream()
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.state->legacy_default_stream;
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
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_event(
                ch,
                (CUevent)args->imms.cuevent.get(),
                std::move(args->caps.generic));
        });
}

core::future<std::shared_ptr<clt::CublasHandle>>
clt::Context::cublas_create()
{
    METHOD(cublas_create);

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_CUBLAS_CREATE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then_check_cublas_response()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_cublas_handle(
                *this,
                ch,
                (cublasHandle_t)args->imms.handle.get(),
                std::move(args->caps.generic));
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
            return this->module_load_data(mem)
                .then([contents, mem=std::move(mem)](auto& fut) mutable {
                    return fut.get();
                });
        })
        .unwrap();
}

core::future<std::shared_ptr<clt::Module>>
clt::Context::module_load_data(core::cap::memory& contents)
{
    METHOD(module_load_data);
    LOG_REQ(method)
        << " contents=" << core::to_string(contents);

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MODULE_LOAD_DATA)
        .set_cap(&msg::request::caps::continuation, resp)
        .set_cap(&msg::request::caps::contents, contents)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_module(
                ch,
                (CUmodule)args->imms.cumodule.get(),
                std::move(args->caps.generic));
        });
}

core::future<void>
clt::Context::memcpy_async(core::cap::memory& src, core::cap::memory& dst, Stream& stream)
{
    CUstream custream = stream.get_stream();

    METHOD(memcpy_async);
    LOG_REQ(method)
        << " src=" << core::to_string(src)
        << " dst=" << core::to_string(dst)
        << " custream=" << (void*)custream;

    auto& pimpl = impl::Context::get(*this);
    auto self = pimpl.state->self.lock();

    if (!src.has_all_perms(core::cap::perm_type::PERM_RD)) {
        LOG(ERROR) << "Cannot read source memory capability for copy";
        throw core::cap_perm_error();
    }

    if (!dst.has_all_perms(core::cap::perm_type::PERM_WR)) {
        LOG(ERROR) << "Cannot write to destination memory capability for copy";
        throw core::cap_perm_error();
    }

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MEMCPY_ASYNC)
        .set_imm(&msg::request::imms::custream, (uint64_t)custream)
        .set_cap(&msg::request::caps::src, src)
        .set_cap(&msg::request::caps::dst, dst)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
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
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_SYNCHRONIZE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
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
    ss << "cuda::Context(" << (void*)obj.cucontext << ")";
    return ss.str();
}
