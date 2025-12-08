// standard includes
#include <stdio.h>
#include <cuda_runtime.h>

// my includes
#include "../include/my_utils.hpp"

__global__ void vector_add(const float* a, const float* b, float* c, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}



void run_vector_add(unsigned int n, unsigned int block_size){
    size_t bytes = n * sizeof(float);   // yeah the number of bytes for n, not sure what yet

    // These are arrays to be stored on the host. the CPU
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialise the host arrays with 1s and 2s
    for (int i = 0; i < n; i++){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;

    // Create space on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Move the values of host arrays to device
    // I think its useful to read these commands right to left
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Ok, now the grid definition, this is the thing to really fluently master IMO
    dim3 block(
        block_size  // x
    );
    dim3 grid(
        ceil_div(n, block.x)   // x
    );

    // Perform the vector_add
    vector_add<<<grid, block>>>(d_a, d_b, d_c, n);

    // The host MUST wait until all CUDA work is finished
    cudaDeviceSynchronize();

    // Ok, now we're finished, bring the computed work back
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);   // read right to left

    // Check correctness
    for (int i=0; i<5; i++){
        printf("%f ", h_c[i]);
    }
}


int main(){
    unsigned int n = 1 << 24;    // Bit shifting i.e. ~16M
    unsigned int block_size = getenv("BLOCK_SIZE") ? atoi(getenv("BLOCK_SIZE")) : 256;
    run_vector_add(n, block_size);
    return 0;
}