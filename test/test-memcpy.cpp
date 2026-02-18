#include <filesystem>
#include <cstring>

#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>


using namespace fractos;


void reset_buffers(auto& cpu1, auto& cpu2, auto& gpu1, auto& gpu2, int n_bytes, auto ctx, auto stream) {
    std::memset((void*)cpu1.get_addr(), 0, n_bytes);
    std::memset((void*)cpu2.get_addr(), 0, n_bytes);
    ctx->memset(gpu1->get_deviceptr(), (uint8_t)0, n_bytes, *stream).get();
    ctx->memset(gpu2->get_deviceptr(), (uint8_t)0, n_bytes, *stream).get();
    ctx->synchronize().get();
}

void fill_cpu_buffer(int* cpu_buf, int n_ints, int offset = 0) {
    for (int i = 0; i < n_ints; i++) {
        cpu_buf[i] = i + offset;
    }
}

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
    auto stream = ctx->stream_create(static_cast<CUstream_flags>(1)).get();
    auto legacy_default_stream = ctx->get_legacy_default_stream().get();

    // Set up memory buffer sizes
    int n_bytes = 1 << 12;
    int n_ints = n_bytes / sizeof(int);

    // Allocate cpu mem
    int* cpu_buf1 = (int*)malloc(n_bytes);
    auto cpu_mem1 = ch->make_memory(cpu_buf1, n_bytes).get();
    int* cpu_buf2 = (int*)malloc(n_bytes);
    auto cpu_mem2 = ch->make_memory(cpu_buf2, n_bytes).get();

    // Allocate gpu mem
    auto gpu_mem1 = ctx->mem_alloc(n_bytes).get();
    auto gpu_mem2 = ctx->mem_alloc(n_bytes).get();

    {
        LOG(INFO) << "Testing basic memcpy...";
        reset_buffers(cpu_mem1, cpu_mem2, gpu_mem1, gpu_mem2, n_bytes, ctx, legacy_default_stream);
        fill_cpu_buffer(cpu_buf1, n_ints);
        ctx->memcpy_async(cpu_mem1, gpu_mem1->get_cap_mem(), *stream).get();
        ctx->synchronize().get();
        ctx->memcpy_async(gpu_mem1->get_cap_mem(), cpu_mem2, *stream).get();
        ctx->synchronize().get();

        for (int i = 0; i < n_ints; i++) {
            CHECK(cpu_buf1[i] == i);
        }
    }

    {
        LOG(INFO) << "Testing memcpy in stream...";
        reset_buffers(cpu_mem1, cpu_mem2, gpu_mem1, gpu_mem2, n_bytes, ctx, legacy_default_stream);
        fill_cpu_buffer(cpu_buf1, n_ints);

        // Block stream
        auto gpu_flag = ctx->mem_alloc(256).get();
        ctx->memset(gpu_flag->get_deviceptr(), (uint8_t)0, 256, *legacy_default_stream).get();
        ctx->synchronize().get();
        auto aux_stream = ctx->stream_create(static_cast<CUstream_flags>(1)).get();
        stream->wait_value_32(gpu_flag->get_deviceptr(), 1, 0).get();

        // Add async stream operations
        ctx->memcpy_async(cpu_mem1, gpu_mem1->get_cap_mem(), *stream).get();
        ctx->memcpy_async(gpu_mem1->get_cap_mem(), cpu_mem2, *stream).get();
        // TODO [ra2520] add some more async stream operations like kernel launches

        // Unblock stream
        aux_stream->write_value_32(gpu_flag->get_deviceptr(), 1, 0).get();
        stream->synchronize().get();

        // Check results
        for (int i = 0; i < n_ints; i++) {
            CHECK(cpu_buf2[i] == i);
        }

        aux_stream->destroy().get();
        gpu_flag->destroy().get();
    }

    {
        LOG(INFO) << "Testing parallel memcpy...";
        reset_buffers(cpu_mem1, cpu_mem2, gpu_mem1, gpu_mem2, n_bytes, ctx, legacy_default_stream);
        fill_cpu_buffer(cpu_buf1, n_ints);
        fill_cpu_buffer(cpu_buf2, n_ints, 512);

        // Create extra stream for parallel memcpy
        auto stream2 = ctx->stream_create(static_cast<CUstream_flags>(1)).get();

        // Block streams
        auto gpu_flag = ctx->mem_alloc(256).get();
        ctx->memset(gpu_flag->get_deviceptr(), (uint8_t)0, 256, *legacy_default_stream).get();
        ctx->synchronize().get();
        auto aux_stream = ctx->stream_create(static_cast<CUstream_flags>(1)).get();
        stream->wait_value_32(gpu_flag->get_deviceptr(), 1, 0).get();
        stream2->wait_value_32(gpu_flag->get_deviceptr(), 1, 0).get();

        // Add memcpy to each stream
        ctx->memcpy_async(cpu_mem1, gpu_mem1->get_cap_mem(), *stream).get();
        ctx->memcpy_async(cpu_mem2, gpu_mem2->get_cap_mem(), *stream2).get();

        // Unblock streams
        aux_stream->write_value_32(gpu_flag->get_deviceptr(), 1, 0).get();
        stream->synchronize().get();
        stream2->synchronize().get();

        // Block streams
        stream->wait_value_32(gpu_flag->get_deviceptr(), 1, 0).get();
        stream2->wait_value_32(gpu_flag->get_deviceptr(), 1, 0).get();

        // Add memcpy to each stream
        ctx->memcpy_async(gpu_mem1->get_cap_mem(), cpu_mem2, *stream).get();
        ctx->memcpy_async(gpu_mem2->get_cap_mem(), cpu_mem1, *stream2).get();

        // Unblock streams
        aux_stream->write_value_32(gpu_flag->get_deviceptr(), 1, 0).get();
        stream->synchronize().get();
        stream2->synchronize().get();

        // Check results
        for (int i = 0; i < n_ints; i++) {
            CHECK(cpu_buf1[i] == i + 512);
            CHECK(cpu_buf2[i] == i);
        }

        stream2->destroy().get();
        gpu_flag->destroy().get();
        aux_stream->destroy().get();
    }

    free(cpu_buf1);
    free(cpu_buf2);
    gpu_mem1->destroy().get();
    gpu_mem2->destroy().get();
    stream->destroy().get();
    ctx->destroy().get();
    dev->destroy().get();

    LOG(INFO) << "test done";
}
