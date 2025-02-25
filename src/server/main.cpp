#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <glog/logging.h>
#include <signal.h>

#include <fractos/service/compute/cuda.hpp>

#include "../library/srv_service.hpp"

// #include <../library/common.hpp>

using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace std::chrono_literals;
using namespace fractos::core;



int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    //////////////////////////////////////////////////
    // 1) Parse command line

    /*
     * We let the server choose the remote service name, which should match the
     * one set when starting the client. This name is used to publish the
     * connection request to the GNS.
     */

    auto odesc = common::cmdline::options();
    odesc.add_options()
        ("service-name", common::cmdline::po::value<std::string>()->default_value("service"),
         "name used to publish the service in GNS");
        // ("control-thread", common::cmdline::po::value<std::string>()->value_name("CPUSET")
        //  ->default_value("-0"),
        //  "CPU set to pin the control thread to (see fractos::common::cpu::parse_set())")
        // ("service-threads", common::cmdline::po::value<std::string>()->value_name("CPUSET")
        //  ->default_value("+all"),
        //  "CPU set to create and pin service threads (see fractos::common::cpu::parse_set())")
        // ;
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);

    // Retrieve additional command line arguments
    auto name = args["service-name"].as<std::string>();
    // auto control_thread = common::cpu::parse_set(args["control-thread"].as<std::string>());
    // auto control_cpu = control_thread->pop_front();
    // auto service_cpus = common::cpu::parse_set(args["service-threads"].as<std::string>());

    // LOG(INFO) << "================================================== start "
    // << "service" << service_cpus->size()
    // << "control" << control_thread->size();

    // Log process state when receiving SIGUSR1
    common::signal::init_log_handler(SIGUSR1, ch->get_process());
    LOG(INFO) << "name ch " << ch->get_name();

    // Log current process state
    ch->get_process()->log_state();

    auto srv = test::gpu_device_service::factory();

    LOG(INFO) << "Create service";

    LOG(INFO) << ch;
    LOG(INFO) << "Register service";
    srv->register_service(ch).get();

    auto gns = core::gns::make_service();

    //auto srv = gpu::gpu_device_service::make_device_service_to_gns(*gns, ch, name);


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
    
    auto srv_published = gns->publish_named(ch, srv->req_make_cuda_device, name)
        .get();

    //(void)gns->publish_named(ch, srv->req_get_virtual_device, "get_vdev");

    LOG(INFO) << "channel running";

    //ch->run_until([](){ return false; });
    ch->run_until([srv]() {return srv->exit_requested(); });


    

    // //////////////////////////////////////////////////
    // // 2) Create service object and threads

    // auto srv = impl::make_service(name);
    
    // LOG(INFO) << "================================================== start ";

    // // Create background helper thread that translates signals into requests to
    // // cleanly stop the server
    // auto intr_handler = common::signal::make_handler_thread(
    //     {SIGINT, SIGQUIT},
    //     [srv, ch](auto signum) {
    //         LOG(INFO) << "exit requested ...";
    //         // mark request to exit
    //         srv->request_exit();
    //         // ensure channel::run_until() checks exit request
    //         ch->break_run();
    //         return false;
    //     },
    //     control_cpu);
    // intr_handler.detach();

    // // Service threads
    // std::vector<std::thread> service_threads;
    // for (size_t i = 0; i < service_cpus->size(); i++) {
    //     service_threads.emplace_back([ch, srv, service_cpus]() {
    //                                      common::cpu::pin(service_cpus->pop_front());
    //                                      ch->run_until([srv](){ return srv->exit_requested(); });
    //                                      LOG(INFO) << "======================== end here ";
    //                                  });
    // }

    // // Pin main thread into control cpus
    // common::cpu::pin(control_cpu);

    // //////////////////////////////////////////////////
    // // 3) Publish service object

    // auto full_name = get_name(name);

    // auto gns = core::gns::make_service();

    // auto req_connect =  srv->register_methods(ch)
    //     .get();
    // LOG(INFO) << "================================================== middle";

    

    // auto srv_published = gns->publish_named(ch, req_connect, full_name)
    //     .get();

    // //////////////////////////////////////////////////
    // // 4) Wait for service threads to exit

    // for (auto& thread : service_threads) {
    //     thread.join();
    // }

    LOG(INFO) << "================================================== finish";



    return 0;
}

