// standard includes
#include <stdio>

// cuda inclues
#include <cuda_runtime.h>

// my includes
#include "../include/my_utils.hpp"



// kernel
__global__ void tiledMM_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
){
    // Do matmul
}



// running code
void tiledMM_naive_run(){
}



// entry
int main(int argc, char* argv[]){
    // block size
    // tile size
    tiledMM_naive_run();
}