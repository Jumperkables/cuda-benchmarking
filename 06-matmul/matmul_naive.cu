// standard includes
#include<algorithm>
#include<iostream>
#include<string>
#include<vector>

// CUDA includes
#include<cuda_runtime.h>

// my includes
#include "../include/my_utils.hpp"


/* Naive matrix multiplication kernel
Calculate each element of the output matrix one at a time here
- C[i][j]
*/
__global__ void matmul_naive(
    const float* __restrict__ A,    // MxK
    const float* __restrict__ B,    // KxN
    float* __restrict__ C,          // MxN
    const unsigned int M,    // rows of the first matrix - used for overlaunch guarding
    const unsigned int N,    // columns of the second matrix - used for overlaunch guarding
    const unsigned int K     // The shared dimension of the first and last matrix
){
    // Overlaunch guards
    int col_idx = (blockDim.x * blockIdx.x) + threadIdx.x;  // (0, N)
    int row_idx = (blockDim.y * blockIdx.y) + threadIdx.y;  // (0, M)
    if (col_idx >= N || row_idx >= M) return;

    // Run matmul
    float acc = 0.0f;
    int A_base_row = K * row_idx;
    for (int k=0; k<K; k++){
        acc +=
            A[A_base_row + k] * // Row major => all contiguous in A from the base row
            B[(k*N) + col_idx]; // Row major => we need to keep jumping forward by steps of N for the fixed column index
    }
    C[(row_idx * N) + col_idx] = acc;
}


// runner function
void matmul_naive_run(const int block_dim_x, const int block_dim_y) {
    unsigned int M = 4000;
    unsigned int K = 2000;
    unsigned int N = 3000;
    // define host arrays   C = A * B
    // Row-major
    std::vector<float> h_A(M*K);    // A : MxK
    std::vector<float> h_B(K*N);    // B : KxN
    std::vector<float> h_C(M*N);    // C : MxN
    // Initialise host arrays
    std::fill(h_A.begin(), h_A.end(), 1.0f);
    std::fill(h_B.begin(), h_B.end(), 1.0f);

    // Device arrays
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    int bytes_A = M*K * sizeof(float);
    int bytes_B = K*N * sizeof(float);
    int bytes_C = M*N * sizeof(float);
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), bytes_C, cudaMemcpyHostToDevice);

    // Figure out block and grid sizes
    dim3 block(
        block_dim_x,
        block_dim_y
    );
    dim3 grid(
        ceil_div(N, block.x),   // Think CAREFULLY here, and cross reference with my indexing in the abve kernel code
        ceil_div(M, block.y)    // Think CAREFULLY here, and cross reference with my indexing in the abve kernel code
    );

    // Launch the kernel
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // Try catching runtime errors
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(e));
    }

    // Stop execution until kernel fully returns
    cudaDeviceSynchronize();  // wait

    // Copy back the results array
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);   // <- always read these right to left, helpes me a lot

    // sanity check
    // Check all values are K (should happen in matmul of matrices all ones)
    bool all_K = std::all_of(h_C.begin(), h_C.end(),
        [K](float x){
            return std::fabs(x- static_cast<float>(K) ) < 1e-5f;
        }
    );
    if (all_K) {
        std::cout << "Success\n";
    }

    // Free device mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// entry
int main(int argc, char* argv[]){
    if (argc != 3){
        std::cout << "Usage: " << argv[0] << " <blockDim.x> <blockDim.y>\n";
        return 1;
    }
    int block_dim_x = std::stoi(argv[1]);
    int block_dim_y = std::stoi(argv[2]);
    matmul_naive_run(block_dim_x, block_dim_y);
}