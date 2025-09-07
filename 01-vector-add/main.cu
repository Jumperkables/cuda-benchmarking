#include <chrono>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>


#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)

// GPU kernel: out[i] = x[i] + y[i]
__global__ void vecAdd_gpu(int n, const float* x, const float* y, float* out){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}


// add together the elements of two arrays
void vecAdd_cpu(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}


// helper function to check what the max error of the array are
float calcMaxError(int n, float *y) {
    float maxError = 0.0f;
    for (int i = 0; i < n; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    return maxError;
}


int main(void) {

    ///////////////////////////////////////////////////////////////////
    // CPU version
    int N = 1<<30;
    float *x = new float[N];    // 1 million, but done using bit shifting binary number (remember Tom that 2**10 ~= 1000)
    float *y = new float[N];

    // instantiate the two arrays on the 'host'
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run the kernel on the 1M elements on the CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    vecAdd_cpu(N, x, y);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = t2 - t1;
    std::cout << "CPU time: " << cpu_ms.count() << " ms\n";

    // check for errors
    float maxError_cpu = calcMaxError(N, y);
    std::cout << "Max error (CPU): " << maxError_cpu << std::endl;
    ///////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    // GPU version
    // allocate and copy device buffers for the device code objects
    size_t bytes = N * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CK(cudaMalloc(&d_x, bytes));
    CK(cudaMalloc(&d_y, bytes));
    CK(cudaMalloc(&d_out, bytes));
    CK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));

    // Define the block size and grid size for this operation
    int blockSize = 1024;    // I should vary this
    int gridSize = (N + blockSize - 1) / blockSize;

    // CUDA timing event setup
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    CK(cudaEventRecord(start)); // Start timing the event?
    vecAdd_gpu<<<gridSize, blockSize>>>(N, d_x, d_y, d_out);   // the actual GPU call
    CK(cudaGetLastError());
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));

    // output the elapsed time
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Kernel time: " << ms << "ms\n";

    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));

    // bring the device code result back to CPU (host) to compare them together
    float *z = new float[N];    // Where the CUDA array is going to land
    CK(cudaMemcpy(z, d_out, bytes, cudaMemcpyDeviceToHost));
    float maxError_gpu = calcMaxError(N, z);
    std::cout << "Max error (GPU): " << maxError_gpu << std::endl;

    // CUDA cleanup
    CK(cudaFree(d_x)); CK(cudaFree(d_y)); CK(cudaFree(d_out));
    // cleanup host array
    delete [] x;
    delete [] y;
    delete [] z;

    ///////////////////////////////////////////////////////////////////

    return 0;
}