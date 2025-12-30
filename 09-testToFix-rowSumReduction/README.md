# Test to Fix - Row Sum Reduction
To test my CUDA instincts, I've asked ChatGPT to design inefficienct CUDA kernels, and challenge myself to see whats wrong with them, fixed them, and improve them, and prove it with NCU
- This is the first of these I've done, i expect to be partially right but also miss obvious things. Lets see how I do

## Original from ChatGPT
I compiled both a dynamic fallback, and a faster precompiled version.
- Precompiled:
    - `blockDim.x == 256 && M = 3000 && N = 2048`
- Dynamic fallback:
    - Anything else

### Performance Notes
- Occupancy: ~80%
- DRAM Throughput: ~53%
- Compute Trhoughput: ~27.4%
- Bytes per row (block):
  - My first guess
    - Read: 
      - N from global memory
      - blockDim.x*(blockDim.x+1)/2 from shared memory
    - Write:
      - blockDim.x + blockDim.x*(blockDim.x+1)/2 into shared memory
  - The actual correct answer:
    - Global memory:
      - Global writes = 1 float (`out[row]`)
      - Global reads = N floats
    - Shared memory:
      - 1) After the loading - `s[tx] = acc`
        - Writes: B floats (where B = blockDim.x)
      - 2) Reduction loop:
        - Each stride = `B/2 + B/4 + ... + 1 = (B-1)`
        - `s[tx] += s[tx + stride] -> s[tx] = s[tx] + s[tx+stride]`
          - 2 reads and 1 write
      - Shared writes = `B + (B-1) = 2B - 1`
      - Shared reads = `2*(B-1) = 2B - 2`
  - 

### My inital thoughts:
- Row reduction kernel is a lot of reads with very simple set of floating point maths
    - I think this algorithm should be memory bottlenecked ideally?
- Occupancy is good, latency hiding is operating as best it can
    - Non perfect occupancy liekly from overhead
- Given this, I suspect that mediocre compute and memory throughput come from stalls and wasted cycles on reads
    - Look for bank conflicts
    - Look for poor reading patterns
    - Look for badly ordered dependencies
    - Poor unrolling?
- FLOP/s and FLOP per Byte derivation attempt:

#### Behaviour:
- Each block is assigned its own row
- `for (int j=tx; j<N; j+=blockDim.x)`
  - Load imbalance if the block size is ever not a multiple of 32, but this is typical

#### `for (int j=tx; j<N; j+=blockDim.x) { acc += A[row * N + j]}`
- At first, i suspected that the big inefficiency here was the fact that we switch back and forth between doing reads on A somewhere in global memory, and then ops back and forth
  - `read A -> acc += -> read A -> acc += -> ...`
  - the strict dependency chain is like so:
  - ```
    for (...) {
     r = load(A[i]);
     acc = acc + r; // <- cannot be launched until r returns
    }
    ```
  - However, proper instruction level parallelism would be:
  - ```cpp
    float a0 = A[i];
    float a1 = A[i + stride];
    float a2 = A[i + stride*2];
    float a3 = A[i + stride*3];
    acc = a0 + a1 + a2 + a3;
    ```
  - The tradeoff here is registers, and being careful with the striding to make sure that Istill have coalesced reads.
- So lets see how many accumulators i can add before we begin to get slowdowns again

| Kernel         | Registers | Duration | L1 Throughput | L2 Throughput |
|----------------|-----------|----------|---------------|---------------|
| 1 accumulator  | 12        | 98.72us  | 16.74%        | 24.02%        | 
| 2 acc          | 16        | 95.94us  | 17.45%        | 24.38%        |
| 3 acc          | 18        | 92.96us  | 17.07%        | 23.71%        |
| 5 acc          | 26        | 93.18us  | 17.21%        | 28.17%        |
| 10 acc         | 38        | 91.20us  | 17.3%         | 31.32%        |
- I should probably stop before either overflows happening causing workload imbalance, or register pressure reduces occupancy (which happens about ~42 registers per thread).
- Cool! Increasing instruction-level parallelism really does make things faster

```
Warp stats 1 accumulator:
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        65.91
    Warp Cycles Per Executed Instruction           cycle        66.07
    Avg. Active Threads Per Warp                                31.91
    Avg. Not Predicated Off Threads Per Warp                    29.09
    ---------------------------------------- ----------- ------------
```
```
Warp stats 10 accumulators:
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle        77.57
    Warp Cycles Per Executed Instruction           cycle        77.86
    Avg. Active Threads Per Warp                                31.87
    Avg. Not Predicated Off Threads Per Warp                    28.24
    ---------------------------------------- ----------- ------------
    
Making N 10 times larger increases stall:
    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle       153.51
    Warp Cycles Per Executed Instruction           cycle       153.61
    Avg. Active Threads Per Warp                                31.98
    Avg. Not Predicated Off Threads Per Warp                    31.28
    ---------------------------------------- ----------- ------------
```
- However, increasing N made my throughput go up a lot too. Genuinely many things to balance.


#### The reduction step - EDIT: Optimising the wrong bottleneck
```
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tx < stride) s[tx] += s[tx + stride];
        __syncthreads();
    }

    if (tx == 0) out[row] = s[0];
```
- This reduction is very inefficient
- The goal is to sum all of shared memory into a single value
- To this end, the striding downwards is ok at first
- But notice that when stride gets lower than 32, i.e. 16, then half the threads in the warp become inactive
- Worse, since the thread index itself gets smaller and smaller, we disproportionately have the rest of the block waiting for us, with now massively underutilised warps
- An amout of this is inevitable chatGPT tells me. Fundamentally there comes a pint with such reductions where parallel work collapses, but lets see if we can make it better

```cpp
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
```
- The above change DID make something better. But not by much at all:
```cpp
// Original implementation
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         9.49
SM Frequency                    Ghz         1.39
Elapsed Cycles                cycle      916,259
Memory Throughput                 %        83.37
DRAM Throughput                   %        83.37
Duration                         us       656.90
L1/TEX Cache Throughput           %        22.81
L2 Cache Throughput               %        35.68
SM Active Cycles              cycle   821,289.06
Compute (SM) Throughput           %        11.00
----------------------- ----------- ------------

// Improved
----------------------- ----------- ------------
Metric Name             Metric Unit Metric Value
----------------------- ----------- ------------
DRAM Frequency                  Ghz         9.49
SM Frequency                    Ghz         1.39
Elapsed Cycles                cycle      913,267
Memory Throughput                 %        82.95
DRAM Throughput                   %        82.95
Duration                         us       654.75
L1/TEX Cache Throughput           %        22.79
L2 Cache Throughput               %        35.71
SM Active Cycles              cycle   821,974.80
Compute (SM) Throughput           %        10.70
----------------------- ----------- ------------
```


### Dynamic Performance
- `blockDim.x == 256 && M = 3000 && N = 2047`
    - Carefully note N=2047, mildly smaller, but does not use the precompiled pathway
```
==PROF== Connected to process 75327 (/home/tomw/projects/cuda-benchmarking/09-testToFix-rowSumReduction/original)
==PROF== Profiling "row_sum_bad" - 0: 0%....50%....100% - 8 passes
Passed sanity check!
==PROF== Disconnected from process 75327
[75327] original@127.0.0.1
  row_sum_bad(const float *, float *, int, int) (23989, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.73
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle       74,737
    Memory Throughput                 %        53.21
    DRAM Throughput                   %        53.21
    Duration                         us        53.63
    L1/TEX Cache Throughput           %        32.88
    L2 Cache Throughput               %        22.26
    SM Active Cycles              cycle    60,981.78
    Compute (SM) Throughput           %        27.44
    ----------------------- ----------- ------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance 
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate    
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.                 

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 23,989
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            4.10
    # SMs                                         SM              82
    Threads                                   thread       6,141,184
    Uses Green Context                                             0
    Waves Per SM                                               48.76
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           12
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        80.06
    Achieved Active Warps Per SM           warp        38.43
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 19.94%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (80.1%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      277,724
    Total DRAM Elapsed Cycles        cycle    6,263,808
    Average L1 Active Cycles         cycle    60,981.78
    Total L1 Elapsed Cycles          cycle    5,992,146
    Average L2 Active Cycles         cycle    46,100.54
    Total L2 Elapsed Cycles          cycle    3,511,200
    Average SM Active Cycles         cycle    60,981.78
    Total SM Elapsed Cycles          cycle    5,992,146
    Average SMSP Active Cycles       cycle    58,717.66
    Total SMSP Elapsed Cycles        cycle   23,968,584
    -------------------------- ----------- ------------
```

### Precompiled Performance
- `blockDim.x == 256 && M = 3000 && N = 2048`
```
==PROF== Connected to process 75394 (/home/tomw/projects/cuda-benchmarking/09-testToFix-rowSumReduction/original)
==PROF== Profiling "row_sum_bad" - 0: 0%....50%....100% - 8 passes
Passed sanity check!
==PROF== Disconnected from process 75394
[75394] original@127.0.0.1
  row_sum_bad(const float *, float *, int, int) (24000, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.73
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle       73,716
    Memory Throughput                 %        53.93
    DRAM Throughput                   %        53.93
    Duration                         us        52.90
    L1/TEX Cache Throughput           %        33.80
    L2 Cache Throughput               %        22.47
    SM Active Cycles              cycle    59,323.27
    Compute (SM) Throughput           %        28.00
    ----------------------- ----------- ------------

    OPT   This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance 
          of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate    
          latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.                 

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 24,000
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            4.10
    # SMs                                         SM              82
    Threads                                   thread       6,144,000
    Uses Green Context                                             0
    Waves Per SM                                               48.78
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block           12
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        79.63
    Achieved Active Warps Per SM           warp        38.22
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 20.37%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (79.6%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle      277,616
    Total DRAM Elapsed Cycles        cycle    6,177,792
    Average L1 Active Cycles         cycle    59,323.27
    Total L1 Elapsed Cycles          cycle    5,872,440
    Average L2 Active Cycles         cycle    44,635.73
    Total L2 Elapsed Cycles          cycle    3,463,200
    Average SM Active Cycles         cycle    59,323.27
    Total SM Elapsed Cycles          cycle    5,872,440
    Average SMSP Active Cycles       cycle    57,371.66
    Total SMSP Elapsed Cycles        cycle   23,489,760
    -------------------------- ----------- ------------
```
