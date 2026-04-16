#include <cuda_runtime.h>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>


using namespace fractos;

int
main(int argc, char *argv[])
{
    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());

    int ndevices;
    CHECK(cudaGetDeviceCount(&ndevices) == cudaSuccess);

    std::vector<cudaStream_t> streams;
    for (auto i = 0; i < ndevices; i++) {
        CHECK(cudaSetDevice(i) == cudaSuccess);
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream) == cudaSuccess);
        streams.push_back(stream);
    }

    for (auto& stream : streams) {
        CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    }

    for (auto& stream : streams) {
        CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    }

    LOG(INFO) << "test done";
}
