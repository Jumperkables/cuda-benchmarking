// standard includes
#include <cassert>
#include <iostream>
#include <vector>

// local includes
#include "rowsum.cuh"
#include "../include/my_utils.hpp"

// cuda includes
#include <cuda_runtime.h>

int main(int argc, char* argv[]){
    // arg handling
    if (argc != 4){
        std::cout << "Usage: " << argv[0] << "<blockDim.x> <M> <N>" << std::endl;
        return 1;
    }
    int block_x = std::atoi(argv[1]);
    int M = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);

    // host arrays
    std::vector<float> h_A(M * N, 1.0f);
    std::vector<float> h_out(M);

    // device arrays
    int bytes_A = M*N*sizeof(float);
    int bytes_out = M*sizeof(float);
    float* d_A = nullptr;
    float* d_out = nullptr;

    // moving to device
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_out, bytes_out);
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);

    // Launch kernel
    if (block_x == 256 && M == 3000 && N == 2048){
        // Compile-time optimised
        dim3 grid(ceil_div(3000*2048, 256));
        dim3 block(256);
        row_sum_bad<<<grid, block>>>(d_A, d_out, 3000, 2048);
    }
    else {
        // Dynamic fallback
        dim3 grid(ceil_div(M*N, block_x));
        dim3 block(block_x);
        row_sum_bad<<<grid, block>>>(d_A, d_out, M, N);
    }
    // Try catching runtime errors
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("Launch error: %s\n", cudaGetErrorString(e));
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) printf("Sync error: %s\n", cudaGetErrorString(e));

    // move back to host
    cudaMemcpy(h_out.data(), d_out, bytes_out, cudaMemcpyDeviceToHost);

    // sanity test
    assert(h_out.size() == static_cast<size_t>(M));
    for (int i=0; i<M; i++){
        if (h_out[i] != static_cast<float>(N)){
            std::cerr << "At least one row wrong\n";
            std::abort();
        }
    }
    std::cout << "Passed sanity check!\n";

    // Free the mallocs
    cudaFree(d_A);
    cudaFree(d_out);
}
