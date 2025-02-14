#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <glog/logging.h>
#include <signal.h>
#

#include <fractos/service/compute/cuda.hpp>
#include <../library/service_impl.hpp>

using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);
    //////////////////////////////////////////////////
    // 1) Parse command line

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <controller_config>" << std::endl;
        return 1;
    }

    auto controller = core::parse_controller_config(argv[1]);
    auto proc = core::make_process(controller).get();
    auto ch = proc->make_channel().get();

    auto name = "test-service-cuda";

    //////////////////////////////////////////////////
    // 2) Get service object, as registered by the server

    LOG(INFO) << "==================================================";

    auto gns = core::gns::make_service();
    auto srv = service::compute::cuda::make_service(ch, *gns, name).get();

    LOG(INFO) << "==================================================";


    return 0;
}

