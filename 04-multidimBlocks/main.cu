#include <cuda_runtime.h>
#include <iostream>


__global__ void MatAdd(float* A, float* B, float* C, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N){
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

int main() {
    int numBlocks = 1;                              // Number of blocks the CUDA kernel will launch
    int N = 32;                                     // The intended size of each dimension of the array
    size_t bytes = N * N * sizeof(float);           // The amount of memory that will need reserving for a single array on the device

    // Host arrays
    float h_A[N * N];
    float h_B[N * N];
    float h_C[N * N];
    for (int i = 0; i < (N*N); i++){
        h_A[i] = 1;
        h_B[i] = 2;
        h_C[i] = 5;
    }

    // Device arrays
    float *d_A, *d_B, *d_C;                               // The null pointers initialised before the memory is allocated
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Run the kernel
    dim3 threadsPerBlock(N,N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result of C back to the host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    }
    cudaDeviceSynchronize();

    // Print result
    std::cout << "Result matrix C:\n";
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            std::cout << h_C[row * N + col] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}