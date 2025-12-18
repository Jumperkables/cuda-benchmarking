# Mental Model for Understanding Memory Throughput - L1, L2, DRAM, and thread indexing 
By analysing my naive GEMM benchmark properly, the aim here is to crystallise a strong mental model of:
- What exactly is going on in memory
- Why the kernel design of naive matmul leads to this
- What kinds of kernel code design decisions will mitigate this in the future

```
==PROF== Profiling "matmul_naive" - 0: 0%....50%....100% - 8 passes
Success
==PROF== Disconnected from process 7377
[7377] matmul_naive@127.0.0.1
  matmul_naive(const float *, const float *, float *, unsigned int, unsigned int, unsigned int) (188, 250, 1)x(16, 16, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
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

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 47,000
    Registers Per Thread             register/thread              30
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread      12,032,000
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                               95.53
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            8
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        99.57
    Achieved Active Warps Per SM           warp        47.79
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle  64,276,502.67
    Total DRAM Elapsed Cycles        cycle  3,077,068,800
    Average L1 Active Cycles         cycle  37,633,714.52
    Total L1 Elapsed Cycles          cycle  3,087,336,942
    Average L2 Active Cycles         cycle  37,251,308.25
    Total L2 Elapsed Cycles          cycle  1,789,352,448
    Average SM Active Cycles         cycle  37,633,714.52
    Total SM Elapsed Cycles          cycle  3,087,336,942
    Average SMSP Active Cycles       cycle  37,580,112.65
    Total SMSP Elapsed Cycles        cycle 12,349,347,768
    -------------------------- ----------- --------------
```

```cpp
__global__ void matmul_naive(
    const float* __restrict__ A,    // MxK
    const float* __restrict__ B,    // KxN
    float* __restrict__ C,          // MxN
    const unsigned int M,    // rows of the first matrix - used for overlaunch guarding
    const unsigned int N,    // columns of the second matrix - used for overlaunch guarding
    const unsigned int K     // The shared dimension of the first and last matrix
){
    // Overlaunch guards
    int col_idx = (blockDim.x * blockIdx.x) + threadIdx.x;  // (0, N)
    int row_idx = (blockDim.y * blockIdx.y) + threadIdx.y;  // (0, M)
    if (col_idx >= N || row_idx >= M) return;

    // Run matmul
    float acc = 0.0f;
    int A_base_row = K * row_idx;
    for (int k=0; k<K; k++){
        acc +=
            A[A_base_row + k] * // Row major => all contiguous in A from the base row
            B[(k*N) + col_idx]; // Row major => we need to keep jumping forward by steps of N for the fixed column index
    }
    C[(row_idx * N) + col_idx] = acc;
}
```

## 1) Blocks -> Warps -> Threads: Indexing and L1/L2 hits
To understand how and why we end up hitting L1 or L2 more, start by thinking very carefully around the naive matmul kernel, and when a warp is created, which memory locations will it be hitting:

### Warp thread construction:
For my `16x16` block size, which `(row_idx, col_idx)` pair get created?
- `threadIdx.x ∈ [0..15]` and `threadIdx.y ∈ [0..15]`
- CUDA assigns threads in ROW-MAJOR order
- `linear_tid = threadIdx.y * blockDim.x + threadIdx.x`
    - Row 0  (y=0): `linear_tid 0-15`
    - Row 1  (y=1): `linear_tid 16-31`
    - ...
    - Row 15 (y=15): `linear_tid 240-255`
- So a warp is now 32 consecutive `linear_tid`s

So, given the above, lets map this onto the co-ordinates of `A` and `B` that are fetched:
- Warp 0:
    - `threadIdx.y ∈ [0,1]` and `threadIdx.x ∈ [0,...,15]`
    - `col_idx = (blockDim.x * blockIdx.x) + threadIdx.x   // (0, N)`
    - `row_idx = (blockDim.y * blockIdx.y) + threadIdx.y   // (0, M)`
    - Each thread executes ALL of the following `for k ∈ [0, 2000]`:
      - `A[(K * row_idx) + k]`
      - `B[(k*N) + col_idx]`
      - `A[(K * {(blockDim.y * blockIdx.y) + threadIdx.y} ) + k]`
      - `B[(k*N) + {(blockDim.x * blockIdx.x) + threadIdx.x}]`
    - All memory addresses asked for by warp 0:
      - `Thread 0:`
        - `threadIdx.y = 0 & threadIdx.x = 0`
        - `A[0, 1, 2, ..., 1999]`
        - `B[0, 3000, 6000, ..., 5997000]`
      - `Thread 1:`
        - `threadIdx.y = 0 & threadIdx.x = 1`
        - `A[0, 1, 2, ..., 1999]`
        - `B[1, 3001, 6001, ..., 5997001]`
      - `Thread 2:`
        - `threadIdx.y = 0 & threadIdx.x = 2`
        - `A[0, 1, 2, ..., 1999]`
        - `B[2, 3002, 6002, ..., 5997002]`
      - `...`
      - `Thread 15:`
        - `threadIdx.y = 0 & threadIdx.x = 15`
        - `A[0, 1, 2, ..., 1999]`
        - `B[15, 3015, 6015, ..., 5997015]`
      - `Thread 16:`
        - `threadIdx.y = 1 & threadIdx.x = 0`
        - `A[2000, 2001, 2002, ..., 3999]`
        - `B[0, 3000, 6000, ..., 5997000]`
      - `...`
      - `Thread 31`
        - `threadIdx.y = 1 & threadIdx.x = 15`
        - `A[2000, 2001, 2002, ..., 3999]`
        - `B[15, 3015, 6015, ..., 60015]`
      - So warp 0 loads:
        - `for each k`:
          - 2 addresses for A
          - 16 addresses for B
        - `A` = `32 threads * K * 2` = `64,000` loads
        - `B` = `32 threads * K * 2` = `64,000` loads
        - BUT `4000 unique addresses for A`
        - `32,000 unique addresses for B`

#### Insights I'm ready for - 2025-12-15
- This kernel does a huge amount of instructions per thread in a warp.
- Many loads, 128K, and a bunch of them on unique addresses
- This probably causes `L1` to work very hard
- However, notice the code ordering:
  - `memReq -> compute -> memReq -> computer -> ...`
  - This looks to memory as `request... pause... request... pause... request...`
  - instead of: `request request request`
  - Generally for memory, burst reads are preferred because:
    - Hard can combine transactions
    - Can saturate the bandwidth
Per warp:
    - `memory level parallelism is low`
Across warps:
    - `latency is hidden by switching`
    - But requests are still spread out in time
    - No burst pressure on L2 and DRAM

