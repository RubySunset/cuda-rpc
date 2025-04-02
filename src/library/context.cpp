
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>

#include <context_impl.hpp>
#include <module_impl.hpp>
#include <memory_impl.hpp>
#include <stream_impl.hpp>


// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Context_impl& Context_impl::get(Context& obj)
{
    return *reinterpret_cast<Context_impl*>(obj._pimpl.get());
}

inline
const Context_impl& Context_impl::get(const Context& obj) 
{
    return *reinterpret_cast<Context_impl*>(obj._pimpl.get());
}




Context::Context(std::shared_ptr<void> pimpl, wire::endian::uint32_t value) : 
    _pimpl(pimpl) {


    DLOG(INFO) << "initialize context : " << value;
}

Context::~Context() {
    DLOG(INFO) << "Context: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}




core::future<void> Context::make_memory_rpc_test(
                    uint64_t size) {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    using msg = ::service::compute::cuda::wire::Context::make_memory_rpc_test;

    DVLOG(logging::SERVICE) << "Context::make_memory <-";
    auto& pimpl = Context_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

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

core::future<std::shared_ptr<Memory>> Context::make_memory(
                    uint64_t size) {
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();
    
    using msg = ::service::compute::cuda::wire::Context::make_memory;

    DVLOG(logging::SERVICE) << "Context::make_memory <-";
    auto& pimpl = Context_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            char* tmp = reinterpret_cast<char*>(args->imms.address.get());

            // get Device object
            std::shared_ptr<Memory_impl> pimpl_(
                new Memory_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.destroy), 
                    false, tmp, size, std::move(args->caps.memory)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Memory> res(new Memory{pimpl, size});
            return res;
        });
}


core::future<std::shared_ptr<Stream>> Context::make_stream(
                CUstream_flags stream_flags, fractos::wire::endian::uint32_t id) {

    using msg = ::service::compute::cuda::wire::Context::make_stream;

    DVLOG(logging::SERVICE) << "Context::make_stream <-";
    auto& pimpl = Context_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            std::shared_ptr<Stream_impl> pimpl_(
                new Stream_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.synchronize),
                        std::move(args->caps.destroy), id}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Stream> res(new Stream{pimpl, flag, id});
            return res;
        });
}


// core::future<std::shared_ptr<Module>> Context::make_module_file(
// // core::future<void> Context::make_module_file(
//             const std::string& file_name) {

//     using msg = ::service::compute::cuda::wire::Context::make_module_file;

//     DVLOG(logging::SERVICE) << "Context::make_module_file <-";

//     auto& pimpl = Context_impl::get(*this);

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
//             wire::error_raise_exception_maybe(args->imms.error);

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


core::future<std::shared_ptr<Module>> Context::make_module_data(
            core::cap::memory& contents, uint64_t module_id) {

    using msg = ::service::compute::cuda::wire::Context::make_module_data;

    DVLOG(logging::SERVICE) << "Context::make_module_data <-";

    auto& pimpl = Context_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get Module object
            std::shared_ptr<Module_impl> pimpl_(
                new Module_impl{{}, ch, args->imms.error,
                        std::move(args->caps.get_function),
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Module> res(new Module{pimpl, module_id});
            return res;
        });
}




core::future<void> Context::synchronize() {
    using msg = ::service::compute::cuda::wire::Context::synchronize;

    DVLOG(logging::SERVICE) << "Context::synchronize <-";

    auto& pimpl = Context_impl::get(*this);

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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

core::future<void> Context::destroy() {
    using msg = ::service::compute::cuda::wire::Context::destroy;

    DVLOG(logging::SERVICE) << "Context::destroy <-";

    auto& pimpl = Context_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_ctx_destroy)
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
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

