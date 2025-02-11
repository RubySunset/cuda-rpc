#include <fractos/service/compute/cuda.hpp>
#include <./service_impl.hpp>

using namespace fractos::service::compute::cuda;


core::future<std::shared_ptr<Service>>
make_service(std::shared_ptr<core::channel> ch,
             core::gns::service& gns, const std::string& name,
             const std::chrono::microseconds& wait_time = std::chrono::seconds{0})
{
    CHECK(false);
}

core::future<std::shared_ptr<Service>>
make_service(std::shared_ptr<core::channel> ch,
             core::cap::request& service_req)
{
    CHECK(false);
}


Service::~Service()
{
    auto& pimpl = impl::Service::get(*this);
    pimpl.destroy()
        .get();
}

core::future<std::shared_ptr<Device>>
Service::make_device(uint64_t device_id)
{
    auto& pimpl = impl::Service::get(*this);
    CHECK(false);
}

core::future<void>
Service::destroy()
{
    auto& pimpl = impl::Service::get(*this);
    if (not pimpl.destroy_sent.test_and_set()) {
        CHECK(false);
    }
}
