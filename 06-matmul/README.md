# 06 Matrix Multiplication
Now to move onto the bigger guns. I'll start with a naive implementation of a simpler case to tst my CUDA coding skills

## TODO before moving on
- Vary block size
- Vary `K`
- Try pragma unroll
- And note the changes

## Important notes for future Tom
- IF YOU SEE `BLAS` YOU MUST CHECK CAREFULLY. Probs column major.
  - Most standard C++ packages and column major
  - includes `BLAS` and `cuBLAS`
  - Otherwise, probably going to have been written in column major format
- Don't forget to catch CUDA runtime errors

## Naive Implementation
- `A: M x K`
- `B: K x N`
- `C: M x N`
- Compute `C = A * B`
- Naive: Each thread computes a single element of `C`
  - Each thread will need to access the entire respective row and column of `A` and `B`
  - As far as I can tell right now, its mostly just going to be memory overhead
  - 
### Benchmarking - Naive:
```cpp
// Run matmul
float acc = 0.0f;
int A_base_row = K * row_idx;
for (int k=0; k<K; k++){
    acc +=
        A[A_base_row + k] * // Row major => all contiguous in A from the base row
        B[(k*N) + col_idx]; // Row major => we need to keep jumping forward by steps of N for the fixed column index
}
C[(row_idx * N) + col_idx] = acc;
```
- `FLOPs ~= 2 * M * K * N`
  - From `acc += ..` and `A[..] + B[..]`
- `GFLOP/s`

## Misc Learning:
- Row Major notation `MxN`:
  - I always forget which way around this is
  - Row major `MxN` means `M` rows
  - Think about it compuationally `X[r][c]` would be correctly indexed reading left to right like this
- Thinking very carefully about what indexes rows and columns:
    - At first, this may seem counter-intuitive:
      - `int col_idx = (blockIdx.x * blockDim.x) + threadIdx.x;`
      - Why `x` for columns, isn't that backwards?
      - No it isn't backwards, draw it down and think about how you'll actually be indexing and moving
- `std::vector<std::vector<float>>` on the host side is tempting
  - **DO NOT DO THIS**
    - Each row is a separate allocation
    - Memory is not contiguous
    - data() doesn't give a flat buffer