#include <fractos/service/compute/cuda.hpp>
#include <cuda.h>

using namespace fractos::service::compute::cuda;

/****************************************************************************
 *
 * CUdevice
 */
Device::~Device()
{
    auto& pimpl = impl::Device::get(*this); // 
    pimpl.destroy()
        .get();
}

core::future<std::shared_ptr<Context>>
Device::make_context(const std::vector<CUctxCreateParams>& params,  unsigned int flags)
{
    auto& pimpl = impl::Device::get(*this); // 
    CHECK(false);
}

core::future<void>
Device::destroy()
{
    auto& pimpl = impl::Device::get(*this); // 
    if (not pimpl.destroy_sent.test_and_set()) {
        CHECK(false);
    }
}

/****************************************************************************
 *
 * CUcontext
 */

Context::~Context()
{
    //
}

core::future<std::shared_ptr<Context>>
Context::make_module(uint64_t device_id)
{
    //
}

core::future<void>
Context::destroy()
{
    //
}
