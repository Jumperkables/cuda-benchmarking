#pragma once

__global__ void row_sum_bad(
    const float* A,
    float* out,
    int M, int N
);