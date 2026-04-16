#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace fractos;

bool is_close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= (tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b))));
}

int main(int argc, char *argv[]) {
    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());

    cublasHandle_t handle;
    CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS);

    const int n = 5;
    const int mat_size = n * n;
    const float alpha = 2.0f;
    const float beta = 0.0f;
    const float one = 1.0f;

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 10.0f);
    std::vector<float> h_A(mat_size, 1.0f);
    std::vector<float> h_B(mat_size, 2.0f);
    std::vector<float> h_C(mat_size, 0.0f);
    std::vector<float> h_res(mat_size);

    float *d_x, *d_y, *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_x, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_y, n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_A, mat_size * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_B, mat_size * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(&d_C, mat_size * sizeof(float)) == cudaSuccess);

    LOG(INFO) << "Testing Level 1: SAXPY";
    CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_res.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < n; ++i) {
        float expected = alpha * h_x[i] + h_y[i];
        CHECK(is_close(h_res[i], expected));
    }

    // TODO [ra2520] add this test back once we add the ability to send results back to the host
    // LOG(INFO) << "Testing Level 1: SDOT";
    // float dot_res;
    // CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    // CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    // CHECK(cublasSdot(handle, n, d_x, 1, d_y, 1, &dot_res) == CUBLAS_STATUS_SUCCESS);
    // CHECK(cudaDeviceSynchronize() == cudaSuccess);
    // float expected_dot = 0.0f;
    // for(int i=0; i<n; ++i) expected_dot += h_x[i] * h_y[i];
    // CHECK(is_close(dot_res, expected_dot));

    LOG(INFO) << "Testing Level 2: SGEMV";
    CHECK(cudaMemcpy(d_A, h_A.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_A, n, d_x, 1, &beta, d_y, 1) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_res.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < n; ++i) {
        CHECK(is_close(h_res[i], n * alpha)); 
    }

    LOG(INFO) << "Testing Level 3: SGEMM";
    CHECK(cudaMemcpy(d_A, h_A.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_B, h_B.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_A, n, d_B, n, &beta, d_C, n) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_res.data(), d_C, mat_size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < mat_size; ++i) {
        CHECK(is_close(h_res[i], n * alpha * 2.0f)); 
    }

    LOG(INFO) << "Testing Extension: SGEAM (Matrix Transpose)";
    for(int i=0; i<mat_size; ++i) h_A[i] = (float)i;
    CHECK(cudaMemcpy(d_A, h_A.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &one, d_A, n, &beta, d_B, n, d_C, n) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_res.data(), d_C, mat_size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            CHECK(is_close(h_res[j * n + i], h_A[i * n + j])); 
        }
    }

    // ==================================================================================
    // EXTENDED TESTS
    // ==================================================================================

    LOG(INFO) << "Testing Extension: GemmEx (Mixed Precision Support)";
    // Reset inputs for GemmEx
    std::fill(h_A.begin(), h_A.end(), 1.0f);
    std::fill(h_B.begin(), h_B.end(), 2.0f);
    CHECK(cudaMemcpy(d_A, h_A.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_B, h_B.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    
    // Using CUDA_R_32F for input/output and Compute Type to match SGEMM behavior
    CHECK(cublasGemmEx(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n,
                       &alpha,
                       d_A, CUDA_R_32F, n,
                       d_B, CUDA_R_32F, n,
                       &beta,
                       d_C, CUDA_R_32F, n,
                       CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_res.data(), d_C, mat_size * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < mat_size; ++i) {
        CHECK(is_close(h_res[i], n * alpha * 2.0f));
    }

    // ----------------------------------------------------------------------------------
    // Batched Data Setup
    // ----------------------------------------------------------------------------------
    const int batch_count = 3;
    const long long int stride = mat_size; // Stride in elements
    size_t batch_mem_size = batch_count * mat_size * sizeof(float);

    std::vector<float> h_batch_A(batch_count * mat_size, 1.0f);
    std::vector<float> h_batch_B(batch_count * mat_size, 2.0f);
    std::vector<float> h_batch_C(batch_count * mat_size, 0.0f);
    std::vector<float> h_batch_res(batch_count * mat_size);

    float *d_batch_A, *d_batch_B, *d_batch_C;
    CHECK(cudaMalloc(&d_batch_A, batch_mem_size) == cudaSuccess);
    CHECK(cudaMalloc(&d_batch_B, batch_mem_size) == cudaSuccess);
    CHECK(cudaMalloc(&d_batch_C, batch_mem_size) == cudaSuccess);

    CHECK(cudaMemcpy(d_batch_A, h_batch_A.data(), batch_mem_size, cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_batch_B, h_batch_B.data(), batch_mem_size, cudaMemcpyHostToDevice) == cudaSuccess);

    // ----------------------------------------------------------------------------------
    
    LOG(INFO) << "Testing Extension: GemmStridedBatchedEx";
    CHECK(cublasGemmStridedBatchedEx(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, n, n,
                                     &alpha,
                                     d_batch_A, CUDA_R_32F, n, stride,
                                     d_batch_B, CUDA_R_32F, n, stride,
                                     &beta,
                                     d_batch_C, CUDA_R_32F, n, stride,
                                     batch_count,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);

    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_batch_res.data(), d_batch_C, batch_mem_size, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < batch_count * mat_size; ++i) {
        CHECK(is_close(h_batch_res[i], n * alpha * 2.0f));
    }

    // ----------------------------------------------------------------------------------

    LOG(INFO) << "Testing Extension: GemmBatchedEx (Pointer Array)";
    // Prepare arrays of pointers
    std::vector<float*> h_ptrs_A(batch_count);
    std::vector<float*> h_ptrs_B(batch_count);
    std::vector<float*> h_ptrs_C(batch_count);

    for(int i = 0; i < batch_count; ++i) {
        h_ptrs_A[i] = d_batch_A + (i * stride);
        h_ptrs_B[i] = d_batch_B + (i * stride);
        h_ptrs_C[i] = d_batch_C + (i * stride);
    }

    // Allocate pointer arrays on device
    float **d_ptrs_A, **d_ptrs_B, **d_ptrs_C;
    CHECK(cudaMalloc(&d_ptrs_A, batch_count * sizeof(float*)) == cudaSuccess);
    CHECK(cudaMalloc(&d_ptrs_B, batch_count * sizeof(float*)) == cudaSuccess);
    CHECK(cudaMalloc(&d_ptrs_C, batch_count * sizeof(float*)) == cudaSuccess);

    CHECK(cudaMemcpy(d_ptrs_A, h_ptrs_A.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_ptrs_B, h_ptrs_B.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(d_ptrs_C, h_ptrs_C.data(), batch_count * sizeof(float*), cudaMemcpyHostToDevice) == cudaSuccess);

    // Reset C for verification
    CHECK(cudaMemset(d_batch_C, 0, batch_mem_size) == cudaSuccess);

    CHECK(cublasGemmBatchedEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              n, n, n,
                              &alpha,
                              (const void * const *)d_ptrs_A, CUDA_R_32F, n,
                              (const void * const *)d_ptrs_B, CUDA_R_32F, n,
                              &beta,
                              (void * const *)d_ptrs_C, CUDA_R_32F, n,
                              batch_count,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);

    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(cudaMemcpy(h_batch_res.data(), d_batch_C, batch_mem_size, cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < batch_count * mat_size; ++i) {
        CHECK(is_close(h_batch_res[i], n * alpha * 2.0f));
    }

    // Clean up extended test resources
    CHECK(cudaFree(d_batch_A) == cudaSuccess);
    CHECK(cudaFree(d_batch_B) == cudaSuccess);
    CHECK(cudaFree(d_batch_C) == cudaSuccess);
    CHECK(cudaFree(d_ptrs_A) == cudaSuccess);
    CHECK(cudaFree(d_ptrs_B) == cudaSuccess);
    CHECK(cudaFree(d_ptrs_C) == cudaSuccess);

    // Standard cleanup
    CHECK(cudaFree(d_x) == cudaSuccess);
    CHECK(cudaFree(d_y) == cudaSuccess);
    CHECK(cudaFree(d_A) == cudaSuccess);
    CHECK(cudaFree(d_B) == cudaSuccess);
    CHECK(cudaFree(d_C) == cudaSuccess);
    CHECK(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);

    LOG(INFO) << "CUBLAS unit tests passed successfully.";
    return 0;
}
