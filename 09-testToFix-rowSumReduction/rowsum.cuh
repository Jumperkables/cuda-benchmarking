#pragma once

__global__ void row_sum_bad(
    const float* A,
    float* out,
    int M, int N
);

__global__ void row_sum_better_reduction(
    const float* A,
    float* out,
    int M, int N
);

__global__ void row_sum_acc2(
    const float* A,
    float* out,
    int M, int N
);

__global__ void row_sum_acc3(
    const float* A,
    float* out,
    int M, int N
);

__global__ void row_sum_acc5(
    const float* A,
    float* out,
    int M, int N
);

__global__ void row_sum_acc10(
    const float* A,
    float* out,
    int M, int N
);