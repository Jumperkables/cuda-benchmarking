#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include "../include/my_utils.hpp"



// SAXPY Kernel
// const to tell compiler and readers the value won't be changing
// __restrict__ to tell the compiler that x, y, and s shouldn't be overlapping
__global__ void saxpy(
    const float a,
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ s,
    const int n
) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;
    float xi = x[idx];
    float yi = y[idx];
    s[idx] = (a * xi) + yi;
}



// main function
int main () {
    // Initialise host arrays and constants
    unsigned int n = 1<<28;
    std::vector<float> h_x(n); // std::vectors for memory safety and C++ work
    std::vector<float> h_y(n);
    std::vector<float> h_s(n);
    float a = 2.5f; // Don't forget to add f to the end, not doing so leave is a double. very inefficient
    // practice using for loops
    for (int i=0; i<n; i++){
        h_x[i] = 1.0f;
    }
    // practice using std::fill
    std::fill(h_y.begin(), h_y.end(), 2.0f);

    // Define block and grid size
    unsigned int block_size = 256;
    dim3 block(block_size);
    dim3 grid(ceil_div(n, block.x));

    // Initialise device arrays
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_s = nullptr;
    std::size_t bytes = n * sizeof(float);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_s, bytes);

    // Copy device array values
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice);

    // Launch the saxpy kernel
    saxpy<<<grid, block>>>(a, d_x, d_y, d_s, n);

    // Wait for the saxpy kernel
    cudaDeviceSynchronize();

    // Copy memory back over
    cudaMemcpy(h_s.data(), d_s, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_s);

    // Sanity check
    std::size_t to_print = std::min<std::size_t>(10, h_s.size());

    // First up to 10 elements
    for (std::size_t i = 0; i < to_print; ++i) {
        printf("h_s[%zu] = %f\n", i, h_s[i]);
    }

    // Last up to 10 elements
    std::size_t start = h_s.size() - to_print;  // safe even if n < 10
    for (std::size_t i = start; i < h_s.size(); ++i) {
    printf("h_s[%zu] = %f\n", i, h_s[i]);
    }
}