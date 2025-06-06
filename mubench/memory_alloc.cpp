#include <cuda_runtime.h>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <glog/logging.h>
#include <vector>

#include <./util-cmdline.hpp>

using namespace fractos;
namespace cmdline = fractos::common::cmdline;
using namespace std::chrono;


int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    auto odesc = options();
    odesc.add_options()
        ("buffer-size", cmdline::po::value<size_t>()->required(),
         "buffer size")
        ("mode", cmdline::po::value<std::string>()->required(),
         "mode of operation: <cuda|fractos>")
        ;
    auto [args, pch, output, metric, control_thread, measurement_threads] = parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, pch->get_process());

    auto num_measurement_threads = measurement_threads->size();
    auto buffer_size = args["buffer-size"].as<size_t>();
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

    size_t allocs_size = 1000000;
    struct state_type {
        std::shared_ptr<service::compute::cuda::Context> ctx;
        size_t alloc_index;
        std::vector<void*> allocs_cuda;
        std::vector<std::shared_ptr<service::compute::cuda::Memory>> allocs_fractos;

        std::shared_ptr<core::channel> ch;
    };

    std::shared_ptr<service::compute::cuda::Service> srv;
    std::shared_ptr<service::compute::cuda::Device> dev;

    switch (mode) {
    case CUDA:
        break;
    case FRACTOS:
        auto gns = core::gns::make_service();
        srv = service::compute::cuda::make_service(
            pch, *gns, "fractos::service::compute::cuda").get();
        srv->init(0).get();
        dev = srv->device_get(0).get();
        break;
    }

    auto cleanup_cuda = [](auto& conn) {
        for (size_t i = 0; i < conn->alloc_index; i++) {
            auto& elem = conn->allocs_cuda[i];
            auto err = cudaFree(elem);
            CHECK(err == cudaSuccess);
            elem = 0;
        }
        conn->alloc_index = 0;
    };

    auto cleanup_fractos = [](auto& conn) {
        for (size_t i = 0; i < conn->alloc_index; i++) {
            auto& elem = conn->allocs_fractos[i];
            if (elem) {
                elem->destroy().get();
                elem = nullptr;
            }
        }
        conn->alloc_index = 0;
    };

    std::mutex conns_mutex;
    std::list<std::shared_ptr<state_type>> conns;

    auto results = exp.run(
        metric,
        // get_connection
        [&](auto thread_idx) {
            auto state = std::make_shared<state_type>();
            state->ch = pch->get_process()->make_channel(pch->get_config())
                .get();
            state->alloc_index = 0;

            switch (mode) {
            case CUDA:
                state->allocs_cuda.resize(allocs_size);
                break;
            case FRACTOS:
                state->ctx = dev->make_context(0).get();
                state->ctx->set_channel(state->ch);
                state->allocs_fractos.resize(allocs_size);
                break;
            }
            {
                auto conns_lock = std::unique_lock(conns_mutex);
                conns.push_back(state);
            }
            return state;
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
            auto idx = conn->alloc_index++;
            switch (mode) {
            case CUDA:
            {
                if (idx == allocs_size) [[unlikely]] {
                    cleanup_cuda(conn);
                    finish_experiment(start_time, true);
                } else {
                    auto err = cudaMalloc(&conn->allocs_cuda[idx], buffer_size);
                    if (err == cudaErrorMemoryAllocation) [[unlikely]] {
                        cleanup_cuda(conn);
                        finish_experiment(start_time, true);
                    } else {
                        finish_experiment(start_time);
                    }
                }
                break;
            }
            case FRACTOS:
            {
                if (idx == allocs_size) [[unlikely]] {
                    cleanup_fractos(conn);
                    finish_experiment(start_time, true);
                } else {
                    conn->ctx->make_memory(buffer_size)
                        .then([&finish_experiment, &cleanup_fractos, conn, idx, start_time](auto&& fut) {
                            try {
                                conn->allocs_fractos[idx] = fut.get();
                            } catch (const service::compute::cuda::CudaError& e) {
                                CHECK(e.cuerror == CUDA_ERROR_OUT_OF_MEMORY);
                                cleanup_fractos(conn);
                                finish_experiment(start_time, true);
                                return;
                            }
                            finish_experiment(start_time);
                        })
                        .as_callback();
                }
                break;
            }
            }
        });

    results.write_csv(output);

    for (auto conn : conns) {
        switch (mode) {
        case CUDA:
            cleanup_cuda(conn);
            break;
        case FRACTOS:
            cleanup_fractos(conn);
            break;
        }
    }
}
