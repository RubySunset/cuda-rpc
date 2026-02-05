#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <glog/logging.h>
#include <signal.h>
#include <csignal>
#include <thread>
#include <string>

#include <fractos/service/compute/cuda.hpp>

#include "./cuda_host_cb_manager.hpp"
#include "./service.hpp"

using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace std::chrono_literals;
using namespace fractos::core;



int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    //  Parse command line
    auto odesc = common::cmdline::options();
    odesc.add_options()
        ("service-name", common::cmdline::po::value<std::string>()
         ->default_value("fractos::service::compute::cuda"),
         "name used to publish the service in GNS");

    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);

    // Retrieve additional command line arguments
    auto name = args["service-name"].as<std::string>();
   

    // Log process state when receiving SIGUSR1
    common::signal::init_log_handler(SIGUSR1, ch->get_process());
    DVLOG(logging::SERVICE) << "name ch " << ch->get_name();

    // Log current process state
    ch->get_process()->log_state();

    auto srv = impl::Service::factory();

    // Start cuda host cb thread
    std::thread cuda_host_cb_thread([&]{
        std::string name = "cuda-host-cb-thread";
        if (pthread_setname_np(pthread_self(), name.c_str()) != 0) {
            LOG(WARNING) << "Failed to set cuda host cb thread name";
        }
        CudaHostCBManager& man = get_cuda_host_cb_manager();
        man.set_channel(ch);
        man.run();
    });

    LOG(INFO) << "Create cuda service";
    srv->register_service(ch).get();

    auto gns = core::gns::make_service();

    // Create background helper thread that translates signals into requests to
    // cleanly stop the server.
    auto intr_handler = common::signal::make_handler_thread(
        {SIGINT, SIGQUIT},
        [srv, ch](auto signum) {
            LOG(WARNING) << "exit requested ...";
            // mark request to exit
            srv->request_exit();
            // ensure channel::run_until() checks exit request
            ch->break_run();
            return false;
        });
    intr_handler.detach();
    
    auto srv_published = gns->publish_named(ch, srv->req_connect, name)
        .get();


    LOG(INFO) << "channel running";

    ch->run_until([srv]() {return srv->exit_requested(); });

    LOG(INFO) << "Stopping cuda host cb thread";
    get_cuda_host_cb_manager().stop();
    cuda_host_cb_thread.join();

    LOG(INFO) << "================================================== finish cuda service";



    return 0;
}

