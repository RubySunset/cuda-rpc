#include <filesystem>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <numeric>
#include <vector>


using namespace fractos;

int
main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());
    auto gns = core::gns::make_service();

    auto path = std::filesystem::path(argv[0]).parent_path();

    auto srv = service::compute::cuda::make_service(ch, *gns, "fractos::service::compute::cuda").get();
    srv->init(0).get();

    CHECK(srv->device_get_count().get() > 0);
    auto dev = srv->device_get(0).get();
    LOG(INFO) << "device 0: " << service::compute::cuda::to_string(*dev) << " - " << dev->get_name().get();
    CHECK(dev->total_mem().get() > 0);

    auto ctx = dev->make_context(0).get();
    auto mod = ctx->module_load(path / "module.ptx").get();
    auto sum_array = mod->get_function("sum_array").get();

    size_t buffer_size = 10;
    std::vector<int> buffer;
    for (size_t i = 0; i < buffer_size; i++) {
        buffer.push_back(i);
    }
    auto cpu_mem = ch->make_memory(buffer.data(), buffer_size * sizeof(buffer[0])).get();

    auto gpu_mem1 = ctx->mem_alloc(buffer_size * sizeof(buffer[0])).get();
    auto gpu_mem2 = ctx->mem_alloc(sizeof(buffer[0])).get();

    {
        ch->copy(cpu_mem, gpu_mem1->get_cap_mem()).get();
        sum_array->launch(*ctx->get_legacy_default_stream().get(), {1,1,1}, {1,1,1}, gpu_mem1->get_deviceptr(), buffer_size, gpu_mem2->get_deviceptr()).get();
        ctx->synchronize().get();

        auto expected = std::accumulate(buffer.begin(), buffer.end(), 0);
        ch->copy(gpu_mem2->get_cap_mem(), cpu_mem).get();
        CHECK(expected == buffer[0]);
    }

    {
        size_t offset1 = 2;
        size_t offset2 = 2;
        auto tmp_mem = ch->diminish(cpu_mem,
                                    offset1 * sizeof(buffer[0]),
                                    (buffer_size - offset2) * sizeof(buffer[0]),
                                    core::cap::PERM_NONE).get();
        ch->copy(tmp_mem, gpu_mem1->get_cap_mem()).get();
        sum_array->launch(*ctx->get_legacy_default_stream().get(), {1,1,1}, {1,1,1}, gpu_mem1->get_deviceptr(), buffer_size - offset1-offset2, gpu_mem2->get_deviceptr()).get();
        ctx->synchronize().get();

        auto expected = std::accumulate(buffer.begin() + offset1, buffer.end() - offset2, 0);
        ch->copy(gpu_mem2->get_cap_mem(), cpu_mem).get();
        CHECK(expected == buffer[0]);
    }

    gpu_mem1->destroy().get();
    gpu_mem2->destroy().get();
    sum_array->destroy().get();
    mod->destroy().get();
    ctx->destroy().get();
    dev->destroy().get();

    LOG(INFO) << "test done";
}
