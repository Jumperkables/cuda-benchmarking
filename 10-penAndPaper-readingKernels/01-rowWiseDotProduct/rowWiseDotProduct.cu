// Kernel to analyze (pen-and-paper): row-wise dot product
// A and B are row-major: M rows, N columns
__global__ void row_dot_suspicious(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ out,
                                   int M, int N)
{
    int row = blockIdx.x;
    int tx  = threadIdx.x;
    if (row >= M) return;

    float acc = 0.0f;
    int base = row * N;

    // Each thread walks the row with a stride of blockDim.x
    for (int j = tx; j < N; j += blockDim.x) {
        float a = A[base + j];
        float b = B[base + j];
        acc += a * b;
    }

    // Reduce within the block
    __shared__ float s[256];     // assume blockDim.x == 256
    s[tx] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
}
