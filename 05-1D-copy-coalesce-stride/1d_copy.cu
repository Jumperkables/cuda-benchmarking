#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// My includes
#include "../include/my_utils.hpp"



// Cuda kernel
__global__ void copy_1d(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int stride,
    const int n
) {
    // calculate idx
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    idx *= stride
    if (idx >= n) return;   // guard overshot threads

    // Do the copy
    out[idx] = in[idx];
}



// running function
void copy_1d_run(unsigned int block_size, unsigned int stride){
    // create the host arrays:
    unsigned int n = 1 << 30;
    std::vector<float> h_in(n);
    std::vector<float> h_out(n);
    std::fill(h_in.begin(), h_in.end(), 1.0f);
    std::fill(h_out.begin(), h_out.end(), 3.0f);

    //// sanity check
    // first 10
    std::cout << "Before - First 5" << "\n";
    for (int i=0; i<5; i++){
        std::cout << "h_in " << h_in[i] << "\n";
        std::cout << "h_out " << h_out[i] << "\n";
    }
    // last 10
    std::cout << "Before - Last 5" << "\n";
    for (int i=n-5; i<n; i++){
        std::cout << "h_in " << h_in[i] << "\n";
        std::cout << "h_out " << h_out[i] << "\n";
    }

    // create block and grid
    dim3 grid(ceil_div(n, block_size));
    dim3 block(block_size);

    // create device arrays
    std::size_t bytes = n * sizeof(float);
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // copy memory to device arrays
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out.data(), bytes, cudaMemcpyHostToDevice);

    // launch the kernel
    copy_1d<<<grid, block>>>(d_in, d_out, stride, n);

    // wait for kernel to finish executing
    cudaDeviceSynchronize();

    // copy arrays back overs
    cudaMemcpy(h_in.data(), d_in, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // Free CUDA memory
    cudaFree(d_in);
    cudaFree(d_out);

    //// sanity check
    // first 5
    std::cout << "After - First 5" << "\n";
    for (int i=0; i<5; i++){
        std::cout << "h_in" << h_in[i] << "\n";
        std::cout << "h_out" << h_out[i] << "\n";
    }
    // last 5
    std::cout << "After - Last 5" << "\n";
    for (int i=n-5; i<n; i++){
        std::cout << "h_in " << h_in[i] << "\n";
        std::cout << "h_out " << h_out[i] << "\n";
    }
}



// main
int main (int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "Usage: " << argv[0] << "<block_size> <stride>\n";
        return 1;
    }
    unsigned int block_size = std::stoi(argv[1]);
    unsigned int stride = std::stoi(argv[2]);
    copy_1d_run(block_size, stride);
}