#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>

#include <./common.hpp>
#include <context_impl.hpp>
#include <module_impl.hpp>
#include <memory_impl.hpp>
#include <stream_impl.hpp>
#include <event_impl.hpp>


using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
srv::to_string(const srv::Context& obj)
{
    auto& pimpl = impl::Context::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Context& obj)
{
    std::stringstream ss;
    ss << "cuda::Context(" << &obj << ")";
    return ss.str();
}



impl::Context::Context(std::shared_ptr<fractos::core::channel> ch,
                       std::shared_ptr<srv::Device> device,
                       fractos::core::cap::request req_generic,
                       fractos::core::cap::request req_memory,
                       fractos::core::cap::request req_memory_rpc_test,
                       fractos::core::cap::request req_stream,
                       fractos::core::cap::request req_event,
                       fractos::core::cap::request req_module_data,
                       fractos::core::cap::request req_ctx_sync,
                       fractos::core::cap::request req_ctx_destroy)
    :ch(ch)
    ,context(0)
    ,device(device)
    ,req_generic(std::move(req_generic))
    ,req_memory(std::move(req_memory))
    ,req_memory_rpc_test(std::move(req_memory_rpc_test))
    ,req_stream(std::move(req_stream))
    ,req_event(std::move(req_event))
    ,req_module_data(std::move(req_module_data))
    ,req_ctx_sync(std::move(req_ctx_sync))
    ,req_ctx_destroy(std::move(req_ctx_destroy))
    ,context_ptr(new char[512])
{
    const_cast<CUcontext&>(context) = (CUcontext)context_ptr.get();
}

srv::Context::Context(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t value)
    :_pimpl(pimpl)
{
}

srv::Context::~Context()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
}


CUcontext
srv::Context::get_context() const
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.context;
}

core::future<unsigned int>
srv::Context::get_api_version()
{
    METHOD(Context, get_api_version);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Context::OP_GET_API_VERSION)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.version.get();
        });
}

core::future<size_t>
srv::Context::get_limit(CUlimit limit)
{
    METHOD(Context, get_limit);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv::wire::Context::OP_GET_LIMIT)
        .set_imm(&msg::request::imms::limit, limit)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.value.get();
        });
}

std::shared_ptr<srv::Device>
srv::Context::get_device()
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.device.lock();
}




core::future<void>
srv::Context::make_memory_rpc_test(uint64_t size)
{
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    using msg = ::service::compute::cuda::wire::Context::make_memory_rpc_test;

    DVLOG(logging::SERVICE) << "Context::make_memory <-";
    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_memory_rpc_test)
        .set_imm(&msg::request::imms::size, size) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([size, t_start](auto& fut) {
            auto [ch, args] = fut.get();

            
            
            auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);
            LOG(INFO) << "time for make_memory_rpc_test server sync at client: " << t_usec.count() << std::endl;


            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for  Context::make_memory");
                DVLOG(logging::SERVICE) << "Context::make_memory ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::make_memory ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // char* tmp = reinterpret_cast<char*>(args->imms.address.get());

            // // get Device object
            // std::shared_ptr<Memory_impl> pimpl_(
            //     new Memory_impl{{}, ch, args->imms.error, 
            //             std::move(args->caps.destroy), 
            //         false, tmp, size, std::move(args->caps.memory)}
            //     );
            // pimpl_->self = pimpl_;
            // auto pimpl = static_pointer_cast<void>(pimpl_);
            // std::shared_ptr<Memory> res(new Memory{pimpl, size});
            // return res;
        });
}

core::future<std::shared_ptr<srv::Memory>>
srv::Context::make_memory(uint64_t size)
{
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();
    
    using msg = ::service::compute::cuda::wire::Context::make_memory;

    DVLOG(logging::SERVICE) << "Context::make_memory <-";
    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_memory)
        .set_imm(&msg::request::imms::size, size) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([size, t_start](auto& fut) {
            auto [ch, args] = fut.get();

            
            auto t_usec = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_start);

            LOG(INFO) << "time for make_memory server sync at client: " << t_usec.count() << std::endl;

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for  Context::make_memory");
                DVLOG(logging::SERVICE) << "Context::make_memory ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::make_memory ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            char* tmp = reinterpret_cast<char*>(args->imms.address.get());

            // get Device object
            auto pimpl_ = std::make_shared<impl::Memory>(
                ch, args->imms.error,
                tmp, size,
                std::move(args->caps.destroy),
                std::move(args->caps.memory));
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Memory> res(new Memory{pimpl, size});
            return res;
        });
}


core::future<std::shared_ptr<srv::Stream>>
srv::Context::make_stream(CUstream_flags stream_flags, fractos::wire::endian::uint32_t id)
{
    using msg = ::service::compute::cuda::wire::Context::make_stream;

    DVLOG(logging::SERVICE) << "Context::make_stream <-";
    auto& pimpl = impl::Context::get(*this);

    unsigned int flag = (unsigned int) stream_flags;
    DLOG(INFO) << "stream flag is " << flag;
    DLOG(INFO) << "stream id is " << id;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_stream)
        .set_imm(&msg::request::imms::flags, flag) // unsigned int vs uint32_t
        .set_imm(&msg::request::imms::stream_id, id) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([flag, id](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for  Context::make_stream");
                DVLOG(logging::SERVICE) << "Context::make_stream ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::make_stream ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            auto pimpl_ = std::make_shared<impl::Stream>(
                ch, args->imms.error,
                id,
                std::move(args->caps.synchronize),
                std::move(args->caps.destroy));
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Stream> res(new Stream{pimpl, flag, id});
            return res;
        });
}


core::future<std::shared_ptr<srv::Event>>
srv::Context::make_event(fractos::wire::endian::uint32_t flags)
{
    using msg = ::service::compute::cuda::wire::Context::make_event;

    DVLOG(logging::SERVICE) << "Context::make_event <-";
    auto& pimpl = impl::Context::get(*this);

    unsigned int flag = flags;
    DVLOG(logging::SERVICE) << "event flag is " << flag;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_event)
        .set_imm(&msg::request::imms::flags, flag) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([flag](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for  Context::make_event");
                DVLOG(logging::SERVICE) << "Context::make_event ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::make_event ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            auto pimpl_ = std::make_shared<impl::Event>(
                ch, args->imms.error,
                // std::move(args->caps.synchronize),std::move(args->caps.record),
                std::move(args->caps.destroy));
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Event> res(new Event{pimpl, flag});
            return res;
        });
}



// core::future<std::shared_ptr<Module>> Context::make_module_file(
// // core::future<void> Context::make_module_file(
//             const std::string& file_name) {

//     using msg = ::service::compute::cuda::wire::Context::make_module_file;

//     DVLOG(logging::SERVICE) << "Context::make_module_file <-";

//     auto& pimpl = impl::Context::get(*this);

//     auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
//     return pimpl.ch->make_request_builder<msg::request>(pimpl.req_module_data)// file
//         // .set_imm(&msg::request::imms::name, name) // unsigned int vs uint32_t
//         .set_imm(&msg::request::imms::file_name_size, file_name.size())
//         .set_imm(offsetof(msg::request::imms, file_name), file_name.c_str(), file_name.size())
//         .set_cap(&msg::request::caps::continuation, resp)
//         .on_channel()
//         .invoke(resp) // wait for srv_handle
//         .unwrap()
//         .then([file_name](auto& fut) { // function_name
//             auto [ch, args] = fut.get();

//             if (not args->has_exactly_args()) {
//                 // throw core::other_error("invalvalue response format for Context::make_module_file");
//                 DVLOG(logging::SERVICE) << "Context::make_module_file ->"
//                 <<" error= OTHER args";
//             }

//             DVLOG(logging::SERVICE) << "Context::make_module_file ->"
//                                     << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
//             fractos::wire::error_raise_exception_maybe(args->imms.error);

//             // get Module object
//             std::shared_ptr<Module_impl> pimpl_(
//                 new Module_impl{{}, ch, args->imms.error,
//                         std::move(args->caps.get_function),
//                         std::move(args->caps.destroy)}
//                 );
//             pimpl_->self = pimpl_;
//             auto pimpl = static_pointer_cast<void>(pimpl_);
//             std::shared_ptr<Module> res(new Module{pimpl, file_name});
//             return res;
//         });
// }


core::future<std::shared_ptr<srv::Module>>
srv::Context::make_module_data(core::cap::memory& contents, uint64_t module_id)
{
    using msg = ::service::compute::cuda::wire::Context::make_module_data;

    DVLOG(logging::SERVICE) << "Context::make_module_data <-";

    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_module_data) // file
         .set_imm(&msg::request::imms::module_id, module_id)
        // .set_imm(offsetof(msg::request::imms, file_name), file_name.c_str(), file_name.size())
        .set_cap(&msg::request::caps::continuation, resp)
        .set_cap(&msg::request::caps::cuda_file, contents)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([module_id](auto& fut) { // function_name
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for Context::make_module_data");
                DVLOG(logging::SERVICE) << "Context::make_module_data ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::make_module_data ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // get Module object
            auto pimpl_ = std::make_shared<impl::Module>(
                ch, args->imms.error,
                std::move(args->caps.generic),
                std::move(args->caps.get_function),
                std::move(args->caps.destroy));
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Module> res(new Module{pimpl, module_id});
            return res;
        });
}

core::future<void>
srv::Context::synchronize()
{
    using msg = ::service::compute::cuda::wire::Context::synchronize;

    DVLOG(logging::SERVICE) << "Context::synchronize <-";

    auto& pimpl = impl::Context::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_ctx_sync)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_sync
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Context::synchronize");
                DVLOG(logging::SERVICE) << "Context::synchronize ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::synchronize ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}

core::future<void>
srv::Context::destroy()
{
    auto& pimpl = impl::Context::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Context::do_destroy()
{
    using msg = ::service::compute::cuda::wire::Context::destroy;

    DVLOG(logging::SERVICE) << "Context::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_ctx_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Context::destroy");
                DVLOG(logging::SERVICE) << "Context::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Context::destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
