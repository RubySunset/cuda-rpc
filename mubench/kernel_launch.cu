#include <cuda_runtime.h>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <glog/logging.h>
#include <vector>

#include <./util-cmdline.hpp>

#include <./empty_kernel.cu>

using namespace fractos;
namespace cmdline = fractos::common::cmdline;
using namespace std::chrono;


int
main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    auto odesc = options();
    odesc.add_options()
        ("ptx", cmdline::po::value<std::string>()->default_value(""),
         "ptx file to execute from")
        ("mode", cmdline::po::value<std::string>()->required(),
         "mode of operation: <cuda|fractos>")
        ;
    auto [args, pch, output, metric, control_thread, measurement_threads] = parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, pch->get_process());

    auto num_measurement_threads = measurement_threads->size();
    auto ptx_path = args["ptx"].as<std::string>();
    auto mode_str = args["mode"].as<std::string>();
    enum mode_type {
        CUDA,
        FRACTOS,
    } mode;
    if (mode_str == "cuda") {
        mode = CUDA;
    } else if (mode_str == "wrapper") {
        mode = CUDA;
    } else if (mode_str == "fractos") {
        mode = FRACTOS;
    } else {
        LOG(FATAL) << "not implemented";
    }

    auto exp = common::experiment::make_experiment(
        num_measurement_threads, *control_thread, *measurement_threads);

    control_thread->pin();

    struct global_state_type {
        std::shared_ptr<service::compute::cuda::Service> srv;
        std::shared_ptr<service::compute::cuda::Device> dev;
    } global;

    struct thread_state_type {
        std::shared_ptr<service::compute::cuda::Context> ctx;
        std::shared_ptr<service::compute::cuda::Module> mod;
        std::shared_ptr<service::compute::cuda::Function> fun;
        std::shared_ptr<core::channel> ch;
    };

    switch (mode) {
    case CUDA:
        break;
    case FRACTOS:
        auto gns = core::gns::make_service();
        global.srv = service::compute::cuda::make_service(
            pch, *gns, "fractos::service::compute::cuda").get();
        global.srv->init(0).get();
        global.dev = global.srv->device_get(0).get();
        break;
    }

    auto results = exp.run(
        metric,
        // get_connection
        [&](auto thread_idx) {
            auto tstate = std::make_shared<thread_state_type>();
            tstate->ch = pch->get_process()->make_channel(pch->get_config())
                .get();

            switch (mode) {
            case CUDA:
                break;
            case FRACTOS:
                tstate->ctx = global.dev->make_context(0).get();
                tstate->ctx->set_channel(tstate->ch);
                tstate->mod = tstate->ctx->module_load(ptx_path).get();
                tstate->fun = tstate->mod->get_function("empty").get();
                break;
            }
            return tstate;
        },
        // run_until
        [&](auto thread_idx, auto& conn, auto& finish_experiment, auto&& stop_cond) {
            conn->ch->run_until(stop_cond);
        },
        // stop_run_until
        [&](auto thread_idx, auto& conn) {
            conn->ch->break_run();
        },
        // start_experiment
        [&](auto thread_idx, auto& conn, auto start_time, auto& finish_experiment) {
            switch (mode) {
            case CUDA:
            {
                empty<<<256, 256>>>();
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
                finish_experiment(start_time);
                break;
            }
            case FRACTOS:
            {
                conn->fun->launch({256, 1, 1}, {256, 1, 1})
                    .then([conn, &finish_experiment, start_time](auto& fut) {
                        conn->ctx->synchronize()
                            .then([&finish_experiment, start_time](auto& fut) {
                                finish_experiment(start_time);
                            })
                            .as_callback();
                    })
                    .as_callback();
                break;
            }
            }
        });

    results.write_csv(output);
}
