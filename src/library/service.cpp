// #pragma once
#include <fractos/service/compute/cuda.hpp>
#include <fractos/core/future.hpp>
#include <./service_impl.hpp>
#include <./common.hpp>

using namespace impl;
using namespace fractos;
// using namespace fractos::service::compute::cuda;


Service_impl::Service_impl(std::string name)
    :_requested_exit(false)
    ,_name(name)
{
}

void
Service_impl::request_exit()
{
    _requested_exit.store(true);
}

bool
Service_impl::exit_requested() const
{
    return _requested_exit.load();
}


std::shared_ptr<impl::Service_impl>
impl::make_service(std::string name)
{
    auto res = std::shared_ptr<impl::Service_impl>(new impl::Service_impl(name));
    res->_self = res;
    return res;
}


core::future<core::cap::request>
Service_impl::register_methods(std::shared_ptr<core::channel> ch)
{
    CHECK(false);
}

// core::future<std::shared_ptr<Service>>
// service::compute::cuda::make_service(std::shared_ptr<core::channel> ch,
//              core::gns::service& gns, const std::string& name,
//              const std::chrono::microseconds& wait_time = std::chrono::seconds{0})
// {
//     auto full_name = get_name(name);

//     LOG_OP("cuda::make_service")
//         << " <-"
//         << " name=" << full_name;

//     // Get the one request we need from the GNS, wait until it has been
//     // published for up to wait_time microseconds.

//     return gns.get_wait_for<core::cap::request>(ch, name, chrono::seconds{0})
//         .then([ch, name](auto& fut) {
//             core::cap::request req;
//                 try {
//                     req = std::move(fut.get()); 
//                     DLOG(INFO) << "Found service";
//                 } catch (const fractos::core::gns::token_error& e) {
//                     LOG(INFO) << "Can't find service";
//                 }

//                 // Build service PIMPL object, with the requests sent by the
//                 // server
//                 std::shared_ptr<Service_impl> pimpl_(
//                     new Service_impl{{}, ch, std::move(req)});
//                 pimpl_->self = pimpl_;
//                 auto pimpl = std::static_pointer_cast<void>(pimpl_);
//                 unique_ptr<Service> res(new Service{pimpl});
//                 res->_pimpl = pimpl; // ? depulcate>

//                 LOG_OP("example::make_service_cuda")
//                     << " ->"
//                     << " " << to_string(*res);

//                 // Return Service object to client.
                
//                 return res;
//             });
// }

// core::future<std::shared_ptr<Service>>
// make_service(std::shared_ptr<core::channel> ch,
//              core::cap::request& service_req)
// {
//     CHECK(false);
// }


// fractos::service::compute::cuda::Service::Service()
//     :_pimpl(Service_impl::make_service(std::string name))
// {
// }


// fractos::service::compute::cuda::Service::Service(std::string name)
//     :_pimpl(Service_impl::make_service(std::string name))
// {
// }







// fractos::service::compute::cuda::Service::Service()
// {
// }

// fractos::service::compute::cuda::Service::~Service()
// {
//     auto& pimpl = Service_impl::get(*this);
//     CHECK(false);
// }





// core::future<std::shared_ptr<Device>>
// Service::make_device(uint64_t device_id)
// {
//     auto& pimpl = Service_impl::get(*this);
//     CHECK(false);
// }

// core::future<void>
// Service::destroy()
// {
//     auto& pimpl = Service_impl::get(*this);
//     if (not pimpl.destroy_sent.test_and_set()) {
//         CHECK(false);
//     }
// }
