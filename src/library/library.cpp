#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include "./library_impl.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Library;
using namespace fractos;


#define IMPL_CLASS impl::Library
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Library>;


std::shared_ptr<clt::Library>
impl::make_library(std::shared_ptr<fractos::core::channel> ch,
                   CUlibrary culibrary,
                   fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::LibraryState>();
    state->req_generic = std::move(req_generic);
    state->culibrary = culibrary;

    return impl::Library::make(ch, state);
}

core::future<void>
impl::LibraryState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto self = this->self.lock();

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
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

CUlibrary
clt::Library::get_library() const
{
    auto& pimpl = impl::Library::get(*this);
    return pimpl.state->culibrary;
}


std::string
clt::to_string(const Library& obj)
{
    auto& pimpl = impl::Library::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const Library& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const LibraryState& obj)
{
    std::stringstream ss;
    ss << "cuda::Library(" << (void*)obj.culibrary << ")";
    return ss.str();
}
