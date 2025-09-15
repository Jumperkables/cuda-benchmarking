#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>

#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// ===== kernels you said you'd use =====
__global__ void clear_coalesced_gridstride(float* __restrict__ v, size_t n){
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += step) v[i] = 0.0f;
}

__global__ void clear_strided_gridstride(float* __restrict__ v, size_t n, int s){
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += step) {
        size_t j = i * (size_t)s;     // scatter by stride s
        if (j < n) v[j] = 0.0f;       // write only if in range
    }
}

// small helper
static inline double gbps(size_t bytes, float ms){
    // bytes / (ms/1e3) / 1e9
    return (double)bytes / (double)ms * 1e-6;
}

int main(int argc, char** argv){
    // Params (override via CLI: N stride)
    size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 28); // 268,435,456 elements (~1.0 GiB)
    int stride = (argc > 2) ? std::atoi(argv[2]) : 32;                          // try 1 (coalesced) vs 32/64 (non-coalesced)

    std::cout << "N = " << N << " elements (" << (N*sizeof(float))/ (1024.0*1024.0*1024.0)
              << " GiB), stride = " << stride << "\n";

    // Device alloc
    float* d_buf = nullptr;
    size_t bytes = N * sizeof(float);
    CK(cudaMalloc(&d_buf, bytes));

    // Launch config (same for both kernels)
    int blockSize = 256;
    // cap grid to something reasonable (max 65535 for legacy 1D grid; you can pick larger with modern launches, but this is fine)
    int gridSize = (int)std::min(
        (size_t)65535,
        (N + (size_t)blockSize - 1) / (size_t)blockSize
    );

    // Warm-up (pay JIT & clock ramp)
    clear_coalesced_gridstride<<<gridSize, blockSize>>>(d_buf, N);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    clear_strided_gridstride<<<gridSize, blockSize>>>(d_buf, N, stride);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // --- Time coalesced ---
    cudaEvent_t s1, e1;
    CK(cudaEventCreate(&s1)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(s1));
    clear_coalesced_gridstride<<<gridSize, blockSize>>>(d_buf, N);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms1 = 0.0f;
    CK(cudaEventElapsedTime(&ms1, s1, e1));
    CK(cudaEventDestroy(s1)); CK(cudaEventDestroy(e1));

    // Effective bytes written: N floats
    double gbps1 = gbps(bytes, ms1);
    std::cout << "Coalesced: " << ms1 << " ms, "
              << gbps1 << " GB/s\n";

    // --- Time strided ---
    // Note: this kernel writes about ceil(N/stride) floats
    size_t written_elems_strided = (N + (size_t)stride - 1) / (size_t)stride;
    size_t bytes_strided = written_elems_strided * sizeof(float);

    cudaEvent_t s2, e2;
    CK(cudaEventCreate(&s2)); CK(cudaEventCreate(&e2));
    CK(cudaEventRecord(s2));
    clear_strided_gridstride<<<gridSize, blockSize>>>(d_buf, N, stride);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e2));
    CK(cudaEventSynchronize(e2));
    float ms2 = 0.0f;
    CK(cudaEventElapsedTime(&ms2, s2, e2));
    CK(cudaEventDestroy(s2)); CK(cudaEventDestroy(e2));

    double gbps2 = gbps(bytes_strided, ms2);
    std::cout << "Strided (s=" << stride << "): " << ms2 << " ms, "
              << gbps2 << " GB/s ("
              << written_elems_strided << " elems written)\n";

    // Cleanup
    CK(cudaFree(d_buf));
    return 0;
}