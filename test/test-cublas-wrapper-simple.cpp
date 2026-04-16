#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <driver_types.h>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>


using namespace fractos;

int
main(int argc, char *argv[]) {
    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());

    LOG(INFO) << "Initialising CUBLAS";
    cublasHandle_t cublas_handle;
    CHECK(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);

    LOG(INFO) << "Setting stream";
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK(cublasSetStream(cublas_handle, stream) == CUBLAS_STATUS_SUCCESS);

    // Test parameters
    const int n = 5;
    float alpha = 2.0f;

    LOG(INFO) << "Creating host arrays";
    float h_x[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_y[n] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    LOG(INFO) << "Creating device arrays";
    float *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, n * sizeof(float)) == cudaSuccess);

    LOG(INFO) << "Copying data to device";
    CHECK(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    LOG(INFO) << "Performing SAXPY operation: y = alpha * x + y";
    CHECK(cublasSaxpy(cublas_handle, n, &alpha, d_x, 1, d_y, 1) == CUBLAS_STATUS_SUCCESS);

    LOG(INFO) << "Synchronizing...";
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    LOG(INFO) << "Copying result back to host";
    CHECK(cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

    // Display results
    LOG(INFO) << "\nSAXPY operation: y = " << alpha << " * x + y";
    LOG(INFO) << "Results:";
    for (int i = 0; i < n; i++) {
      LOG(INFO) << "  y[" << i << "] = " << h_y[i];
    }

    // Cleanup
    CHECK(cudaFree(d_x) == cudaSuccess);
    CHECK(cudaFree(d_y) == cudaSuccess);
    CHECK(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

    LOG(INFO) << "\nCUBLAS test completed successfully!";
}
