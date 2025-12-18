// standard includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

// cuda inclues
#include <cuda_runtime.h>

// my includes
#include "../include/my_utils.hpp"


// Fixed width kernel
template<int BM, int BN, int TK, typename T>
__global__ void tiledMM_naive(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M,
    const int K,
    const int N
){
    // more convenient way to sort shared memory
    __shared__ T As[BM][TK];    // <=== Row major
    __shared__ T Bs[TK][BN];    // <=== Row major

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * BN + tx;
    int row = blockIdx.y * BM + ty;

    T acc = T(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        int a_col = k0 + tx;
        int b_row = k0 + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : T(0);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : T(0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk)
            acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}



// kernel - Dynamic kernel
// Launched when BM and BN are NOT one of the precompiled ones known at compile time
template<typename T>
__global__ void tiledMM_naive_dyn(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int M,
    const int K,
    const int N,
    const int TK
){
    // For the dynamic kernel, invoke shared memory
    //                     As          Bs
    extern __shared__ T shmem[];

    // edge guards
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = tx + (blockIdx.x * blockDim.x);
    int row = ty + (blockIdx.y * blockDim.y);

    // each thread is responsible for exactly 1 element in the output C matrix
    int BM = blockDim.y;
    int BN = blockDim.x;

    // Split between As and Bs subsections
    // Layout: shmem = [As (BM * TK): Bs (TK * BN)]
    T* As = shmem;
    T* Bs = shmem + (BM * TK);

    // 1) Load in the tile_k chunks of A and B - taking advantage of burst memory reads
    // 2) Then do the accumulation and maths afterwards in rapid succession, not waiting for reads
    T acc = T(0);
    for (int k0=0; k0<K; k0+= TK){
        // Load A and B tile chunks into shared memory
        // Load A as [ty, kk] for kk in [0, TK) - ROW MAJOR (TK x BM)
        for (int kk=tx; kk<TK; kk+=BN){
            int a_col = k0 + kk;
            As[(ty*TK) + kk] = (a_col < K) && (row < M) ? A[(K*row)+ a_col] : T(0);   // clever type casting if i do say so myself
        }
        // Load B as - ROW MAJOR (BN x TK) - NOTE INDEXES CAREFULLY
        for (int kk=ty; kk<TK; kk+=BM){
            int b_row = k0 + kk;
            Bs[(kk*BN) + tx] = (b_row < K) && (col < N) ? B[(N*b_row) + col] : T(0);
        }
        __syncthreads();    // waits until all threads are ready probably - looking up exactly what this does tomorrow

        // add everything to the accumulator
        for (int kk=0; kk<TK; kk++){
            acc += As[(ty*TK)+kk] * Bs[(BN*kk)+tx];
        }
        __syncthreads();
    }
    // Now assign the accumulation to the correct position in C
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}



// running code
void tiledMM_naive_run(
    int M, int K, int N,                // Matrix A: MxK  Matric B: KxN
    int block_size_x, int block_size_y, // Block size x and y - also the output C tile dimensions
    int tile_size_K                     // The tile size to load data into and computer over
){
    // Initialise host arrays
    std::vector<float> h_A(M*K);
    std::vector<float> h_B(K*N);
    std::vector<float> h_C(M*N);
    // fill them with some values
    std::fill(h_A.begin(), h_A.end(), 1.0f);
    std::fill(h_B.begin(), h_B.end(), 1.0f);

    // create device arrays
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    int bytes_A = sizeof(float) * M*K;
    int bytes_B = sizeof(float) * K*N;
    int bytes_C = sizeof(float) * M*N;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);   // as always, read from right to left
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // create the block and grid
    dim3 block(
        block_size_x,
        block_size_y
    );
    dim3 grid(
        ceil_div(N, block_size_x),
        ceil_div(M, block_size_y)
    );

    // shared memory
    size_t shmem_bytes = (block.x * tile_size_K + block.y * tile_size_K) * sizeof(float);

    // launch kernel
    // launch pre-cooked if block size aligns
    // else launch specific
    // Fixed launch sizes
    if (block_size_x == 16 && block_size_y == 16 && tile_size_K == 64){
        //           <BM, BN, TK>    <== I learned about templating CUDA kernels 2025-12-15
        tiledMM_naive<16, 16, 64, float><<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    } else if (block_size_x == 32 && block_size_y == 16 && tile_size_K == 32 ) {
        tiledMM_naive<16, 32, 32, float><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    } else{
        tiledMM_naive_dyn<<<grid, block, shmem_bytes>>>(d_A, d_B, d_C, M, K, N, tile_size_K);
    }

    // Try catching runtime errors
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("Launch error: %s\n", cudaGetErrorString(e));
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) printf("Sync error: %s\n", cudaGetErrorString(e));

    // copy back to host
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);

    // sanity check
    // Check all values are K (should happen in matmul of matrices all ones)
    bool all_K = std::all_of(h_C.begin(), h_C.end(),
        [K](float x){
            return std::fabs(x- static_cast<float>(K) ) < 1e-5f;
        }
    );
    if (all_K) {
        std::cout << "Success\n";
    } else{
        std::cout << "Failure\n";
    }

    // Free CUDA mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



// entry
int main(int argc, char* argv[]){
    // arg input checks
    if (argc != 7){
        std::cout << "Usage: " << argv[0] << "<M> <K> <N> <BLOCK_X> <BLOCK_Y> <TILE_K>";
        return 1;
    }

    // matrix dimensions
    int M = std::stoi(argv[1]);
    int K = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);

    // block size - and therefore C output tile size
    int block_size_x = std::stoi(argv[4]);
    int block_size_y = std::stoi(argv[5]);

    // tile size - for the k dimension for reduction
    int tile_size_K = std::stoi(argv[6]);
    tiledMM_naive_run(M, K, N, block_size_x, block_size_y, tile_size_K);
}