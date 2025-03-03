#include <utility>

// #include "./cuda_service.hpp"
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace std;
using namespace fractos::service::compute::cuda;

///////////////////////////////////////////////////////////
// * pimpl pattern
// * all cuda_service_impl related functions
inline
cuda_service_impl& cuda_service_impl::get(cuda_service& obj)
{
    return *reinterpret_cast<cuda_service_impl*>(obj._pimpl.get());
}

inline
const cuda_service_impl& cuda_service_impl::get(const cuda_service& obj)
{
    return *reinterpret_cast<cuda_service_impl*>(obj._pimpl.get());
}

cuda_service::cuda_service(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
    


}


std::shared_ptr<core::channel> cuda_service::get_default_channel() {
    auto& pimpl = cuda_service_impl::get(*this);
    return pimpl.ch;
}

std::shared_ptr<core::channel> cuda_service::get_default_channel() const {
    auto& pimpl = cuda_service_impl::get(*this);
    return pimpl.ch;
}

void cuda_service::set_default_channel(std::shared_ptr<core::channel> ch) {
    auto& pimpl = cuda_service_impl::get(*this);
    pimpl.ch = ch;
}
/////////////////////////////////////////////////////////

/*
 * Make a cuda_service from global_ns which the server publishes to 
 */
core::future<std::unique_ptr<cuda_service>>
fractos::service::compute::cuda::make_cuda_service(fractos::core::gns::service& gns, const std::string& name,
                                    std::shared_ptr<core::channel> ch) {
    
    return gns.get_wait_for<core::cap::request>(ch, name, std::chrono::seconds{0})
        .then([ch, name](auto& fut) {
            core::cap::request req;
                try {
                    req = std::move(fut.get()); 
                    DLOG(INFO) << "Found service";
                } catch (const fractos::core::gns::token_error& e) {
                    LOG(INFO) << "Can't find service";
                }

                std::shared_ptr<cuda_service_impl> pimpl_(
                    new cuda_service_impl{{}, ch, std::move(req)});
                pimpl_->self = pimpl_;
                auto pimpl = std::static_pointer_cast<void>(pimpl_);
                std::unique_ptr<cuda_service> res(new cuda_service{pimpl});

                return res;
              });

} 


/*
 *  Make cuda_device frontend function
 *  makes a request for make_cuda_device and sets the continuation of the response
 */
core::future<std::shared_ptr<cuda_device>> cuda_service::make_cuda_device(uint8_t value) {
    using msg = ::service::compute::cuda::message::cuda_service::make_cuda_device;

    DVLOG(logging::SERVICE) << "cuda_service::make_cuda_device <-";

    auto& pimpl = cuda_service_impl::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_make_cuda_device)
        .set_imm(&msg::request::imms::value, value)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([value](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for cuda_service::make_cuda_device");
                DVLOG(logging::SERVICE) << "cuda_service::make_cuda_device ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_service::make_cuda_device ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get cuda_device object
            std::shared_ptr<cuda_device_impl> pimpl_(
                new cuda_device_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.make_cuda_context),
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<cuda_device> res(new cuda_device{pimpl, value});
            return res;
        });
}


/*
 *  Get published cuda_device
 */
core::future<std::shared_ptr<cuda_device>> cuda_service::get_cuda_device(fractos::core::gns::service& gns, 
                                                                               uint8_t value) {
    using msg = ::service::compute::cuda::message::cuda_service::get_cuda_device;

    DVLOG(logging::SERVICE) << "cuda_service::get_cuda_device <-";

    auto& pimpl = cuda_service_impl::get(*this);

    const string name = "get_vdev";
    auto ch = pimpl.ch;

    return gns.get_wait_for<core::cap::request>(ch, name, chrono::seconds{0})
        .then([ch, name, value](auto& fut) {
            core::cap::request get_vdev = std::move(fut.get());

                auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
                return ch->make_request_builder<msg::request>(get_vdev)
                    .set_imm(&msg::request::imms::value, value)
                    .set_cap(&msg::request::caps::continuation, resp)
                    .on_channel()
                    .invoke(resp)
                    .unwrap()
                    .then([ch, value](auto& fut) {
                        auto [ch, args] = fut.get();

                        if (not args->has_exactly_args()) {
                            // throw core::other_error("invalvalue response format for cuda_service::get_cuda_device");
                            DVLOG(logging::SERVICE) << "cuda_service::get_cuda_device ->"
                                                << " error=OTHER args" ;
                        }

                        DVLOG(logging::SERVICE) << "cuda_service::get_cuda_device ->"
                                                << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
                        wire::error_raise_exception_maybe(args->imms.error);

                        shared_ptr<cuda_device_impl> pimpl_ (
                            new cuda_device_impl{{}, ch, args->imms.error, 
                                    // move(args->caps.allocate_memory),
                                    // move(args->caps.register_function),
                                    move(args->caps.make_cuda_context),
                                    move(args->caps.destroy)}
                            ); 
                        pimpl_->self = pimpl_;
                        auto pimpl = static_pointer_cast<void>(pimpl_);
                        shared_ptr<cuda_device> res(new cuda_device{pimpl, value});
                        return res;
                    });
        })
        .unwrap();
}

