#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <glog/logging.h>
#include <signal.h>
#

#include <fractos/service/compute/cuda.hpp>

using namespace fractos;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    //////////////////////////////////////////////////
    // 2) Get service object, as registered by the server

    LOG(INFO) << "==================================================";

    auto gns = core::gns::make_service();
    auto srv = service::compute::cuda::make_service(ch, *gns, name).get();

    LOG(INFO) << "==================================================";


    return 0;
}

