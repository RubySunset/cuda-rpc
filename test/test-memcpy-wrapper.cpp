#include <cstdint>
#include <cuda.h>
#include <filesystem>
#include <cstring>

#include <fractos/common/logging.hpp>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>

#define CU_CHECK(op)                                              \
    if (CUresult cuerror = (op); cuerror != CUDA_SUCCESS) {       \
        LOG(ERROR) << "CUDA API call failed with error " << cuerror; \
        exit(-1); \
    }

using namespace fractos;


void reset_buffers(void* cpu1, void* cpu2, CUdeviceptr gpu1, CUdeviceptr gpu2, int n_bytes) {
    std::memset(cpu1, 0, n_bytes);
    std::memset(cpu2, 0, n_bytes);
    CU_CHECK(cuMemsetD8(gpu1, 0, n_bytes));
    CU_CHECK(cuMemsetD8(gpu2, 0, n_bytes));
    CU_CHECK(cuCtxSynchronize());
}

void fill_cpu_buffer(uint32_t* cpu_buf, size_t n_ints, size_t offset = 0) {
    for (size_t i = 0; i < n_ints; i++) {
        cpu_buf[i] = i + offset;
    }
}

int
main(int argc, char *argv[])
{
    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());

    LOG(INFO) << "Creating CUDA handles";
    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuCtxCreate(&ctx, NULL, 0, dev))
    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, 0));

    // Set up memory buffer sizes
    size_t n_bytes = 1 << 12;
    size_t n_ints = n_bytes / sizeof(uint32_t);

    // Allocate cpu mem
    uint32_t* cpu_buf1 = (uint32_t*)malloc(n_bytes);
    CUdeviceptr cpu_mem1 = (CUdeviceptr)cpu_buf1;
    uint32_t* cpu_buf2 = (uint32_t*)malloc(n_bytes);
    CUdeviceptr cpu_mem2 = (CUdeviceptr)cpu_buf2;

    LOG(INFO) << "Allocating GPU mem";
    CUdeviceptr gpu_mem1;
    CU_CHECK(cuMemAlloc(&gpu_mem1, n_bytes));
    CUdeviceptr gpu_mem2;
    CU_CHECK(cuMemAlloc(&gpu_mem2, n_bytes));

    {
        LOG(INFO) << "Testing basic memcpy...";
        reset_buffers(cpu_buf1, cpu_buf2, gpu_mem1, gpu_mem2, n_bytes);

        fill_cpu_buffer(cpu_buf1, n_ints);
        CU_CHECK(cuMemcpyAsync(gpu_mem1, cpu_mem1, n_bytes, stream));
        CU_CHECK(cuCtxSynchronize());
        CU_CHECK(cuMemcpyAsync(cpu_mem2, gpu_mem1, n_bytes, stream));
        CU_CHECK(cuCtxSynchronize());

        for (size_t i = 0; i < n_ints; i++) {
            CHECK(cpu_buf2[i] == i);
        }
    }

    {
        LOG(INFO) << "Testing memcpy in stream...";
        reset_buffers(cpu_buf1, cpu_buf2, gpu_mem1, gpu_mem2, n_bytes);
        fill_cpu_buffer(cpu_buf1, n_ints);

        // Block stream
        CUdeviceptr gpu_flag;
        CU_CHECK(cuMemAlloc(&gpu_flag, 256));
        CU_CHECK(cuMemsetD8(gpu_flag, 0, 256));
        CU_CHECK(cuCtxSynchronize());
        CUstream aux_stream;
        CU_CHECK(cuStreamCreate(&aux_stream, 1));
        CU_CHECK(cuStreamWaitValue32(stream, gpu_flag, 1, 0));

        // Add async stream operations
        CU_CHECK(cuMemcpyAsync(gpu_mem1, cpu_mem1, n_bytes, stream));
        CU_CHECK(cuMemcpyAsync(cpu_mem2, gpu_mem1, n_bytes, stream));
        // TODO [ra2520] add some more async stream operations like kernel launches

        // Unblock stream
        CU_CHECK(cuStreamWriteValue32(aux_stream, gpu_flag, 1, 0));
        CU_CHECK(cuStreamSynchronize(stream));

        // Check results
        for (size_t i = 0; i < n_ints; i++) {
            CHECK(cpu_buf2[i] == i);
        }

        CU_CHECK(cuStreamDestroy(aux_stream));
        CU_CHECK(cuMemFree(gpu_flag));
    }

    {
        LOG(INFO) << "Testing parallel memcpy...";
        reset_buffers(cpu_buf1, cpu_buf2, gpu_mem1, gpu_mem2, n_bytes);
        fill_cpu_buffer(cpu_buf1, n_ints);
        fill_cpu_buffer(cpu_buf2, n_ints, 512);

        // Create extra stream for parallel memcpy
        CUstream stream2;
        CU_CHECK(cuStreamCreate(&stream2, 1));

        // Block streams
        CUdeviceptr gpu_flag;
        CU_CHECK(cuMemAlloc(&gpu_flag, 256));
        CU_CHECK(cuMemsetD8(gpu_flag, 0, 256));
        CU_CHECK(cuCtxSynchronize());
        CUstream aux_stream;
        CU_CHECK(cuStreamCreate(&aux_stream, 1));
        CU_CHECK(cuStreamWaitValue32(stream, gpu_flag, 1, 0));
        CU_CHECK(cuStreamWaitValue32(stream2, gpu_flag, 1, 0));

        // Add memcpy to each stream
        CU_CHECK(cuMemcpyAsync(gpu_mem1, cpu_mem1, n_bytes, stream));
        CU_CHECK(cuMemcpyAsync(gpu_mem2, cpu_mem2, n_bytes, stream2));

        // Unblock streams
        CU_CHECK(cuStreamWriteValue32(aux_stream, gpu_flag, 1, 0));
        CU_CHECK(cuStreamSynchronize(stream));
        CU_CHECK(cuStreamSynchronize(stream2));

        // Block streams
        CU_CHECK(cuStreamWaitValue32(stream, gpu_flag, 1, 0));
        CU_CHECK(cuStreamWaitValue32(stream2, gpu_flag, 1, 0));

        // Add memcpy to each stream
        CU_CHECK(cuMemcpyAsync(cpu_mem2, gpu_mem1, n_bytes, stream));
        CU_CHECK(cuMemcpyAsync(cpu_mem1, gpu_mem2, n_bytes, stream2));

        // Unblock streams
        CU_CHECK(cuStreamWriteValue32(aux_stream, gpu_flag, 1, 0));
        CU_CHECK(cuStreamSynchronize(stream));
        CU_CHECK(cuStreamSynchronize(stream2));

        // Check results
        for (size_t i = 0; i < n_ints; i++) {
            CHECK(cpu_buf1[i] == i + 512);
            CHECK(cpu_buf2[i] == i);
        }

        CU_CHECK(cuStreamDestroy(stream2));
        CU_CHECK(cuMemFree(gpu_flag));
        CU_CHECK(cuStreamDestroy(aux_stream));
    }

    free(cpu_buf1);
    free(cpu_buf2);
    CU_CHECK(cuMemFree(gpu_mem1));
    CU_CHECK(cuMemFree(gpu_mem2));
    CU_CHECK(cuStreamDestroy(stream));
    CU_CHECK(cuCtxDestroy(ctx));

    LOG(INFO) << "test done";
}
