# 06 Matrix Multiplication
Now to move onto the bigger guns. I'll start with a naive implementation of a simpler case to tst my CUDA coding skills

## TODO before moving on
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
  - `= 2 * 4000 * **2000** * 3000`
  - `= 48 GFLOP`
  - `t = 26.99ms`
- `GFLOP/s = 1,778 GFLOP/s = 1.78 TFlop/s`

#### Play with grid size:
- `8 x 8`
  - Duration = `28.72ms`
  - Occupancy problem: Max theoretical = 32 = `67% occ`
- `8 x 4`
  - Duration = `54.05ms`
  - Occupancy problem: Max theoretical = 16 = `33% occ`
- `16 x 16`
  - Duration = `26.99ms`
  - Occupancy problem: Max theory = 48 = 100% occ
  - Compute throughput = `97.5%`
  - DRAM throughput = `25.07%`
- `32 x 8`
  - Duration = `27.26ms`
  - 100% occ
- `8 x 32`
  - Duration = `27.90ms`
  - 100% occ

### Lesson in profiling:
```
----------------------- ----------- -------------
Metric Name             Metric Unit  Metric Value
----------------------- ----------- -------------
DRAM Frequency                  Ghz          9.49
SM Frequency                    Ghz          1.39
Elapsed Cycles                cycle    37,680,872
Memory Throughput                 %         97.50
DRAM Throughput                   %         25.07
Duration                         ms         27.01
L1/TEX Cache Throughput           %         97.55
L2 Cache Throughput               %         21.20
SM Active Cycles              cycle 37,633,714.52
Compute (SM) Throughput           %         97.50
----------------------- ----------- -------------
```
```
Compute (SM) Throughput    97.50 %
Memory Throughput          97.50 %
DRAM Throughput            25.07 %
```
- Compute (SM) throughput is NOT % of peak TFLOP/s
  - Actually means, "% of cycles where the SM issued at least one instruction"
  - Includes FP32, INT, address calcs, load/stores, and control flows
  - This is more occupancy than a raw FLOP metric
  - I got this because my occupancy is high
- Memory throughput = 97.5%
  - An aggregation of all of;
    - L1
    - L2
    - Shared memory paths
    - Memory instruction issue rate
  - This is more "how busy were all the memory pipelines"
  - Lots of memory instructions issued, but doesn't mean DRAM was maxxed
- DRAM throughput = 25.07%
  - Another giveaway is `L1 = 97.55% and L2 = 21.20%`
  - We need to slow down here and build another analogy to understand where the L1/L2 latency is going. in [this accompanying md file](memory_throughput_mental_model.md).

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