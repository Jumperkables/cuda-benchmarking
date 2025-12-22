#include "original.cuh"


__global__ void row_sum_bad(const float* __restrict__ A,
                            float* __restrict__ out,
                            int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    if (row >= M) return;

    __shared__ float s[1024];  // still assumes blockDim.x <= 1024

    float acc = 0.0f;
    for (int j = tx; j < N; j += blockDim.x) {
        acc += A[row * N + j];
    }

    s[tx] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}