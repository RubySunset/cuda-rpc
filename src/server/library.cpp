#include <fractos/common/service/srv_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <glog/logging.h>
#include <pthread.h>

#include "./common.hpp"
#include "./library.hpp"


namespace srv = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Library;
using namespace fractos;


std::string
impl::to_string(const impl::Library& obj)
{
    std::stringstream ss;
    ss << "Library(" << (void*)obj.get_remote_culibrary() << ")";
    return ss.str();
}


fractos::core::future<std::tuple<wire::error_type, CUresult, std::shared_ptr<impl::Library>>>
impl::make_library(std::shared_ptr<fractos::core::channel> ch,
                   std::shared_ptr<char[]>& contents,
                   const std::vector<CUjit_option>& jit_options, const std::vector<void*>& jit_values,
                   const std::vector<CUlibraryOption>& lib_options, const std::vector<void*>& lib_values)
{
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;
    std::shared_ptr<Library> res;

    CUlibrary culibrary;
    cuerror = cuLibraryLoadData(&culibrary, contents.get(),
                                (CUjit_option*)jit_options.data(), (void**)jit_values.data(), jit_options.size(),
                                (CUlibraryOption*)lib_options.data(), (void**)lib_values.data(), lib_options.size());
    if (cuerror != CUDA_SUCCESS) {
        return core::make_ready_future(std::make_tuple(error, cuerror, res));
    }

    res = std::make_shared<Library>();
    res->culibrary = culibrary;
    res->contents = contents;

    return ch->make_request_builder<srv_wire_msg::generic::request>(
        ch->get_default_endpoint(),
        [res](auto ch, auto args) {
            res->handle_generic(ch, std::move(args));
        })
        .on_channel()
        .make_request()
        .then([res](auto& fut) {
            res->req_generic = fut.get();
        })
        .then([error, cuerror, res](auto& fut) mutable {
            try {
                fut.get();
            } catch (const core::generic_error& e) {
                error = (wire::error_type)e.error;
            }

            if (error or cuerror) {
                LOG(FATAL) << "TODO: undo Library and return error";
            } else {
                res->self = res;
            }

            return std::make_tuple(error, cuerror, res);
        });
}


CUlibrary
impl::Library::get_remote_culibrary() const
{
    return (CUlibrary)this;
}


void
impl::Library::handle_generic(auto ch, auto args)
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
    CASE_HANDLE(DESTROY, destroy);
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

#undef CASE_HANDLE
}


core::future<std::tuple<wire::error_type, CUresult>>
impl::Library::destroy_maybe(auto ch)
{
    auto self = this->self;
    auto error = wire::ERR_SUCCESS;
    auto cuerror = CUDA_SUCCESS;

    if (not common::service::SrvBase::destroy_maybe()) {
        error = wire::ERR_OTHER;
        return core::make_ready_future(std::make_tuple(error, cuerror));
    }

    return ch->revoke(req_generic)
        .then([ch, this, self](auto& fut) {
            fut.get();

            auto error = wire::ERR_SUCCESS;
            auto cuerror = CUDA_SUCCESS;

            cuerror = cuLibraryUnload(culibrary);
            if (cuerror != CUDA_SUCCESS) {
                goto out_inner;
            }

            out_inner:

            this->self.reset();

            return std::make_tuple(error, cuerror);
        });
}

void
impl::Library::handle_destroy(auto ch, auto args)
{
    METHOD(destroy);
    LOG_REQ(method) << srv::wire::to_string(*args);

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.continuation);
    CHECK_ARGS_EXACT(reqb_cont);

    auto self = this->self;

    return destroy_maybe(ch)
        .then([ch, this, self, args=std::move(args)](auto& fut) {
            auto [error, cuerror] = fut.get();

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
}
