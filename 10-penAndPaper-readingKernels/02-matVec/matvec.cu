// standard includes
#include <iostream>
#include <numeric>
#include <vector>

// local includes
#include "../../include/my_utils.hpp"

// CUDA includes
#include <cuda_runtime.h>



/*
Matrix and vector multiplication
- Generated as inefficient by ChatGPT
- I'm going to add efficiencies to it, diagnose bottlenecks etc...
*/
__global__ void matVec_bad(const float* M, const float* x, float* y, int rows, int cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = tid; r < rows; r += 1)
    {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c)
        {
            float a0 = M[r * cols + c];
            float b0 = x[c];

            sum += a0 * b0;
            y[r] = sum;
        }
    }
}



/*
My improvements involve
- Adding __restrict__ keyword to tensor arguments
    * Improves readability and conveys intent
    * Compiler optimisations allowed
-
*/
__global__ void matVec_improved(
    const float* __restrict__ M,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = tid; r < rows; r += 1)
    {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c)
        {
            float a0 = M[r * cols + c];
            float b0 = x[c];

            float a1 = M[r * cols + c];
            float b1 = x[c];

            sum += a0 * b0;
            sum += (a1 * b1) - (a0 * b0);

            y[r] = sum;
        }
    }
}



int main(int argc, char* argv[]){
    if (argc != 4){
        std::cerr << "Usage: <blockSize> <rows> <cols>\n";
        return 1;
    }
    int blockSize = std::atoi(argv[1]);
    int rows = std::atoi(argv[2]);
    int cols = std::atoi(argv[3]);

    // Host arrays
    std::vector<float> h_M(rows*cols);
    std::vector<float> h_x(cols);
    std::vector<float> h_y(cols);
    std::fill(h_M.begin(), h_M.end(), 1.0f);
    std::fill(h_x.begin(), h_x.end(), 1.0f);
    std::fill(h_y.begin(), h_y.end(), 0.0f);

    // Device arrays
    float* d_M = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    int bytes_M = sizeof(float) * rows * cols;
    int bytes_x = sizeof(float) * cols;
    int bytes_y = sizeof(float) * cols;
    cudaMalloc(&d_M, bytes_M);
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_y, bytes_y);
    cudaMemcpy(d_M, h_M.data(), bytes_M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), bytes_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), bytes_y, cudaMemcpyHostToDevice);

    // Grid and block calculations
    dim3 block(blockSize);
    dim3 grid(ceil_div(rows, blockSize));

    // Launch kernel
    matVec_bad<<<grid, block>>>(d_M, d_x, d_y, rows, cols);

    // Copy back to host
    cudaMemcpy(h_y.data(), d_y, bytes_y, cudaMemcpyDeviceToHost);

    // Print and check correctness
    float sum = std::accumulate(h_y.begin(), h_y.end(), 0.0f);
    std::cout << sum << std::endl;

    // Free da mallocs
    cudaFree(d_M);
    cudaFree(d_x);
    cudaFree(d_y);

    // Return
    return 0;
}