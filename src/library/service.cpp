#include <fractos/service/compute/cuda.hpp>
#include <./service_impl.hpp>
#include <./common.hpp>

using namespace fractos::service::compute::cuda;


core::future<std::shared_ptr<Service>>
make_service(std::shared_ptr<core::channel> ch,
             core::gns::service& gns, const std::string& name,
             const std::chrono::microseconds& wait_time = std::chrono::seconds{0})
{
    auto full_name = get_name(name);

    LOG_OP("cuda::make_service")
        << " <-"
        << " name=" << full_name;

    // Get the one request we need from the GNS, wait until it has been
    // published for up to wait_time microseconds.

    return gns.get_wait_for<core::cap::request>(ch, name, chrono::seconds{0})
        .then([ch, name](auto& fut) {
            core::cap::request req;
                try {
                    req = std::move(fut.get()); 
                    DLOG(INFO) << "Found service";
                } catch (const fractos::core::gns::token_error& e) {
                    LOG(INFO) << "Can't find service";
                }

                // Build service PIMPL object, with the requests sent by the
                // server
                std::shared_ptr<Service_impl> pimpl_(
                    new Service_impl{{}, ch, std::move(req)});
                pimpl_->self = pimpl_;
                auto pimpl = std::static_pointer_cast<void>(pimpl_);
                unique_ptr<Service> res(new Service{pimpl});
                res->_pimpl = pimpl; // ? depulcate>

                LOG_OP("example::make_service_cuda")
                    << " ->"
                    << " " << to_string(*res);

                // Return Service object to client.
                
                return res;
            });
}

core::future<std::shared_ptr<Service>>
make_service(std::shared_ptr<core::channel> ch,
             core::cap::request& service_req)
{
    CHECK(false);
}


Service::~Service()
{
    auto& pimpl = Service_impl::get(*this);
    pimpl.destroy()
        .get();
}

core::future<std::shared_ptr<Device>>
Service::make_device(uint64_t device_id)
{
    auto& pimpl = Service_impl::get(*this);
    CHECK(false);
}

core::future<void>
Service::destroy()
{
    auto& pimpl = Service_impl::get(*this);
    if (not pimpl.destroy_sent.test_and_set()) {
        CHECK(false);
    }
}
