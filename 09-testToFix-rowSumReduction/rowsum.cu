#include "rowsum.cuh"


// Original from ChatGPT
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

/*
- __device__ needed to tell the compiler that this is kernel code
- __inline__ to hint to the compiler to paste this code inline to remove function call overheads
*/
__inline__ __device__ float warp_reduce_sum(float v){
    // ChatGPT made this, and it is AWESOME
    // Full mask for active lanes in the warp
    unsigned mask = 0xffffffffu;
    // Tree reduction within a warp
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v,  8);
    v += __shfl_down_sync(mask, v,  4);
    v += __shfl_down_sync(mask, v,  2);
    v += __shfl_down_sync(mask, v,  1);
    return v;
}


// Better reduction
__global__ void row_sum_better_reduction(
    const float* __restrict__ A,
    float* __restrict__ out,
    int M, int N
)
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

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) { // Changed to leave 1 warps worth of work at the end
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    // Finalise with a single warp
    if (tx < 32) {
        float v = s[tx];
        v = warp_reduce_sum(v);
        if (tx == 0) out[row] = v;
    }
}



// 2 accumulators
__global__ void row_sum_acc2(const float* __restrict__ A,
                            float* __restrict__ out,
                            int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    if (row >= M) return;

    __shared__ float s[1024];  // still assumes blockDim.x <= 1024

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int j = tx; (j+blockDim.x)< N; j += 2*blockDim.x) {
        acc0 += A[row * N + j];
        acc1 += A[row * N + j + blockDim.x];
    }
    float acc = acc0 + acc1;

    s[tx] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}

// 3 accumulators
__global__ void row_sum_acc3(const float* __restrict__ A,
                            float* __restrict__ out,
                            int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    if (row >= M) return;

    __shared__ float s[1024];  // still assumes blockDim.x <= 1024

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j = tx; j+2*blockDim.x< N; j += (3*blockDim.x)) {
        acc0 += A[row * N + j];
        acc1 += A[row * N + j + blockDim.x];
        acc2 += A[row * N + j + blockDim.x*2];
    }
    float acc = acc0 + acc1 + acc2;

    s[tx] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}


// 5 accumulators
__global__ void row_sum_acc5(const float* __restrict__ A,
                            float* __restrict__ out,
                            int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    if (row >= M) return;

    __shared__ float s[1024];  // still assumes blockDim.x <= 1024

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    for (int j = tx; (j+blockDim.x*4)< N; j += 5*blockDim.x) {
        acc0 += A[row * N + j];
        acc1 += A[row * N + j + blockDim.x];
        acc2 += A[row * N + j + blockDim.x*2];
        acc3 += A[row * N + j + blockDim.x*3];
        acc4 += A[row * N + j + blockDim.x*4];
    }
    float acc = acc0 + acc1 + acc2 + acc3 + acc4;

    s[tx] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}


// 10 accumulators
__global__ void row_sum_acc10(const float* __restrict__ A,
                            float* __restrict__ out,
                            int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;

    if (row >= M) return;

    __shared__ float s[1024];  // still assumes blockDim.x <= 1024

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    float acc8 = 0.0f;
    float acc9 = 0.0f;
    for (int j = tx; (j+blockDim.x*9)< N; j += 10*blockDim.x) {
        acc0 += A[row * N + j];
        acc1 += A[row * N + j + blockDim.x];
        acc2 += A[row * N + j + blockDim.x*2];
        acc3 += A[row * N + j + blockDim.x*3];
        acc4 += A[row * N + j + blockDim.x*4];
        acc5 += A[row * N + j + blockDim.x*5];
        acc6 += A[row * N + j + blockDim.x*6];
        acc7 += A[row * N + j + blockDim.x*7];
        acc8 += A[row * N + j + blockDim.x*8];
        acc9 += A[row * N + j + blockDim.x*9];
    }
    float acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9;

    s[tx] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}