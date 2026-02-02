# 02 Matrix Vector Multiplication

- ChatGPT generated the inefficient kernel `matVec_bad`
- I'm going to identify bottlenecks, arithmetic intensity, and other such things, fix them, and make an improved version

## Inefficient
### Code
```cpp
__global__ void matVec_bad(const float* M, const float* x, float* y, int rows, int cols)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = tid; r < rows; r += 1)
    {
        float sum = 0.0f;

        for (int c = 0; c < cols; ++c)
        {
            float a0 = M[r * cols + c];
            float b0 = x[c];
            sum += a0 * b0;
            y[r] = sum;
        }
    }
}
```

### Profile
```
==PROF== Connected to process 25734 (/home/jumperkables/projects/cuda-benchmarking/10-penAndPaper-readingKernels/02-matVec/matvec_bad)
==PROF== Profiling "matVec_bad" - 0: 0%....50%....100% - 8 passes
1e+06
==PROF== Disconnected from process 25734
[25734] matvec_bad@127.0.0.1
  matVec_bad(const float *, const float *, float *, int, int) (40, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           9.49
    SM Frequency                    Ghz           1.39
    Elapsed Cycles                cycle  1,850,456,206
    Memory Throughput                 %          18.16
    DRAM Throughput                   %          15.21
    Duration                          s           1.33
    L1/TEX Cache Throughput           %          49.68
    L2 Cache Throughput               %          18.16
    SM Active Cycles              cycle 460,541,382.56
    Compute (SM) Throughput           %           6.17
    ----------------------- ----------- --------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.08 full
          waves across all SMs. Look at Launch Statistics for more details.                       

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     40
    Registers Per Thread             register/thread              24
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread          10,240
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.08
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 51.22%                                                                    
          The grid for this launch is configured to execute only 40 blocks, which is less than the 82 multiprocessors
          used. This can underutilize some multiprocessors. If you do not intend to execute this kernel concurrently
          with other workloads, consider reducing the block size to have at least one block per multiprocessor or
          increase the size of the grid to fully utilize the available hardware resources. See the Hardware Model
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more
          details on launch configurations.                                                       

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        16.42
    Achieved Active Warps Per SM           warp         7.88
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 83.58%                                                              
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (16.4%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                                   

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ----------------
    Metric Name                Metric Unit     Metric Value
    -------------------------- ----------- ----------------
    Average DRAM Active Cycles       cycle 1,915,527,233.33
    Total DRAM Elapsed Cycles        cycle  151,100,704,768
    Average L1 Active Cycles         cycle   460,541,382.56
    Total L1 Elapsed Cycles          cycle  152,494,581,358
    Average L2 Active Cycles         cycle 1,551,492,889.58
    Total L2 Elapsed Cycles          cycle   87,866,823,744
    Average SM Active Cycles         cycle   460,541,382.56
    Total SM Elapsed Cycles          cycle  152,494,581,358
    Average SMSP Active Cycles       cycle   452,226,105.25
    Total SMSP Elapsed Cycles        cycle  609,978,325,432
    -------------------------- ----------- ----------------

    OPT   Est. Speedup: 18.57%                                                                    
          One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 74.99% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 18.24%                                                                    
          One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum
          instance value is 75.02% above the average, while the minimum instance value is 100.00% below the average.
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 18.57%                                                                    
          One or more L1 Slices have a much lower number of active cycles than the average number of active cycles.
          Maximum instance value is 74.99% above the average, while the minimum instance value is 100.00% below the
          average.                                                                                
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 6.799%                                                                    
          One or more L2 Slices have a much higher number of active cycles than the average number of active cycles.
          Maximum instance value is 8.02% above the average, while the minimum instance value is 1.82% below the
          average.
```

### Diagnosis 
- Launched block size = 256
- rows = 10,000
- columns = 1000
- Duration = 1.33s
- Compute (SM) throughput = 6.17%
- Memory throughput = 18.16%
- Occupancy:
  - 16.42% occupancy
  - 7.88 achieved activate warps per SM
  - Theoretical occupancy is 100%
    - We don't have any hard block limiting problems launching
  - Profiler warns me some SMs have much lower numbers of activate cycles than the average

### Diagnosis Checklist:
1. What SHOULD the bottleneck be
    - Should me a memory bottleneck. The operations are simple matrix mult, the biggest problem should be waiting for matrix elements from memory
    - Therefore, I should optimise for the memory reads foremost
2. Count the read and write bytes
    - From `global` memory
      - Inner loop:
        - 
    - From `shared` memory
      - Shared memory is NOT used at all. This is not inherently wrong, but a red flag for this kind of operation.
        - Array x is constantly re-used. Should consider for shmem


## Improved
```cpp
waiting
```