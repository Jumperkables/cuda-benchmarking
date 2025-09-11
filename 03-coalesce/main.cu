#include <cuda_runtime.h>


__global__ void clear_vector_coalesced(float* v, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)}{
        v[id] = 0.0;    // Coalesced because when a thread comes to access this, it is naturally laid out such that each thread will be taking from contiguous memory
    }
}

__global__ void clear_vector_non_coalesced(float* v, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id *= 4;
    if (id < n)}{
        v[id] = 0.0;
        v[id + 1] = 0.0;
        v[id + 2] = 0.0;
        v[id + 3] = 0.0;
    }
}


int main() {
    clear_vector_coalesced();
}