#include <chrono>
#include <cuda_runtime.h>
#include <iostream>


#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)


__global__ void clear_vector_coalesced(float* v, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n){
        v[id] = 0.0;    // Coalesced because when a thread comes to access this, it is naturally laid out such that each thread will be taking from contiguous memory
    }
}

__global__ void clear_vector_non_coalesced(float* v, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id *= 4;
    if (id < n){
        v[id] = 0.0;
        v[id + 1] = 0.0;
        v[id + 2] = 0.0;
        v[id + 3] = 0.0;
    }
}


int main() {
    // Instantiate vector
    int N = 1<<30;
    float *v = new float[N];
    float *w = new float[N];

    // Instantiate CUDA block and grid size
    int blockSize = 128;   // Multiple of 32
    int gridSize = (N + blockSize - 1) / blockSize;

    // Time the coalesced /////////////////////////////////////////////
    // Allocate and copy device buffers for the device code objects
    size_t bytes = N * sizeof(float);
    float *d_v = nullptr;
    CK(cudaMalloc(&d_v, bytes));
    CK(cudaMemcpy(d_v, v, bytes, cudaMemcpyHostToDevice));

    // Set up timer and run code
    cudaEvent_t start1, stop1;
    CK(cudaEventCreate(&start1));
    CK(cudaEventCreate(&stop1));
    CK(cudaEventRecord(start1));
    clear_vector_coalesced<<<gridSize, blockSize>>>(d_v, N);
    CK(cudaGetLastError());
    CK(cudaEventRecord(stop1));
    CK(cudaEventSynchronize(stop1));

    // output elapse time
    float ms1 = 0.0f;
    CK(cudaEventElapsedTime(&ms1, start1, stop1));
    std::cout << "Coalesced time: " << ms1 << "ms\n";
    CK(cudaEventDestroy(start1));
    CK(cudaEventDestroy(stop1));

    // CUDA Cleanup
    CK(cudaFree(d_v));
    // clean host array
    delete [] v;

    // Time the uncoalesced /////////////////////////////////////////////
    // Allocate and copy device buffers for the device code objects
    float *d_w = nullptr;
    CK(cudaMalloc(&d_w, bytes));
    CK(cudaMemcpy(d_w, w, bytes, cudaMemcpyHostToDevice));

    // Set up timer and run code
    cudaEvent_t start2, stop2;
    CK(cudaEventCreate(&start2));
    CK(cudaEventCreate(&stop2));
    CK(cudaEventRecord(start2));
    clear_vector_non_coalesced<<<gridSize, blockSize>>>(d_w, N);
    CK(cudaGetLastError());
    CK(cudaEventRecord(stop2));
    CK(cudaEventSynchronize(stop2));

    // output elapse time
    float ms2 = 0.0f;
    CK(cudaEventElapsedTime(&ms2, start2, stop2));
    std::cout << "Coalesced time: " << ms2 << "ms\n";
    CK(cudaEventDestroy(start2));
    CK(cudaEventDestroy(stop2));

    // CUDA Cleanup
    CK(cudaFree(d_w));
    // clean host array
    delete [] w;
}