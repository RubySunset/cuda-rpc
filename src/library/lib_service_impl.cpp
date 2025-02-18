#include <./lib_service_impl.hpp>

using namespace fractos;

std::string impl::to_string(const impl::CuService_impl& obj)
{
    return "example::service(" + obj.name + ")";
}



// using namespace impl;

// inline
// std::shared_ptr<Service_impl::service>
// Service_impl::make_service(std::shared_ptr<core::channel> ch,
//     core::gns::service& gns, const std::string& name,
//     const std::chrono::microseconds& wait_time = std::chrono::seconds{0})
// {
//     std::shared_ptr<Service> res(new Service(ch, *gns, name, wait_time));
//     res->self = res;
//     return res;
// }

// inline
// std::shared_ptr<Service_impl>
// Service_impl::make_service(std::string name)
// {
//     auto res = std::shared_ptr<Service_impl>(new Service_impl(name));
//     res->_self = res;
//     return res;
// }

// inline
// Service_impl::Service_impl(std::string name)
//     :name(name)
// {
// }