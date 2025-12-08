# First Kernel
Starting with vector addition

## Profiling
- `benchmarking.py` runs my CUDA implementation with different block sizes

### nsight
- For any arch users reading this, as of 2025-12 you want:
  - `nsight-compute`
  - `nsight-systems`
- `sudo ncu ./vector_add`

#### Naive first run:
- My GPU has 82 streaming multiprocessors
- I only launched one block
- Duration was 3.2e-6
- Ok, occupancy maths time:
  - I'm only launching 1 block
  - My single block contans 256 threads
  - Warps come in 32, so tats 256/32 = 8 warps of work
  - A single streaming multiproc can hold up tp 64 warps
  - So i'm using 8/64 = 1/8 ~ 12.5 ~ 16% efficiency
```
==PROF== Connected to process 48051 (/home/jumperkables/projects/cuda-benchmarking/01_vector-add/vector_add)
==PROF== Profiling "vector_add" - 0: 0%....50%....100% - 8 passes
3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 3.000000 ==PROF== Disconnected from process 48051
[48051] vector_add@127.0.0.1
  vector_add(const float *, const float *, float *, int) (1, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.36
    SM Frequency                    Ghz         1.37
    Elapsed Cycles                cycle        4,396
    Memory Throughput                 %         1.55
    DRAM Throughput                   %         1.55
    Duration                         us         3.20
    L1/TEX Cache Throughput           %        61.08
    L2 Cache Throughput               %         0.98
    SM Active Cycles              cycle        19.65
    Compute (SM) Throughput           %         0.02
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.00 full
          waves across all SMs. Look at Launch Statistics for more details.              

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      1
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread             256
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.00
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 98.78%                                                           
          The grid for this launch is configured to execute only 1 block, which is less than the 82 multiprocessors
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
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        16.38
    Achieved Active Warps Per SM           warp         7.86
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 83.62%                                                     
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (16.4%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle          464
    Total DRAM Elapsed Cycles        cycle      359,424
    Average L1 Active Cycles         cycle        19.65
    Total L1 Elapsed Cycles          cycle      294,966
    Average L2 Active Cycles         cycle       215.94
    Total L2 Elapsed Cycles          cycle      208,848
    Average SM Active Cycles         cycle        19.65
    Total SM Elapsed Cycles          cycle      294,966
    Average SMSP Active Cycles       cycle        19.02
    Total SMSP Elapsed Cycles        cycle    1,179,864
    -------------------------- ----------- ------------
```

#### MORE BLOCKS
- Launch with more blocks, fire everything
- So it turns out that the kernel launch is controlling how many blocks get launched: `vector_add<<<grid, block>>>
- AHA. I had a typo:
  - this was always 1, and not the entire grid as should be intended
```cpp
dim3 grid(
  ceil_div(block_size, block.x)   // x
);
```
- corrected
```cpp
dim3 grid(
  ceil_div(n, block.x)   // x
);
```
- Ok, the below is way better --- in that it also actually finished the job unlike the above
```
==PROF== Connected to process 56099 (/home/jumperkables/projects/cuda-benchmarking/01_vector-add/vector_add)
==PROF== Profiling "vector_add" - 0: 0%....50%....100% - 8 passes
3.000000 3.000000 3.000000 3.000000 3.000000 ==PROF== Disconnected from process 56099
[56099] vector_add@127.0.0.1
  vector_add(const float *, const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle      335,840
    Memory Throughput                 %        91.83
    DRAM Throughput                   %        91.83
    Duration                         us       240.77
    L1/TEX Cache Throughput           %        19.47
    L2 Cache Throughput               %        39.55
    SM Active Cycles              cycle   336,281.33
    Compute (SM) Throughput           %        15.08
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 65,536
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread      16,777,216
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                              133.20
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           16
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        77.36
    Achieved Active Warps Per SM           warp        37.13
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 22.64%                                                     
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (77.4%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle 2,098,474.67
    Total DRAM Elapsed Cycles        cycle   27,421,696
    Average L1 Active Cycles         cycle   336,281.33
    Total L1 Elapsed Cycles          cycle   27,820,002
    Average L2 Active Cycles         cycle      330,716
    Total L2 Elapsed Cycles          cycle   15,949,344
    Average SM Active Cycles         cycle   336,281.33
    Total SM Elapsed Cycles          cycle   27,820,002
    Average SMSP Active Cycles       cycle   334,297.20
    Total SMSP Elapsed Cycles        cycle  111,280,008
    -------------------------- ----------- ------------
```

## Important notes of warp occupancy
A streaming multiprocessor packs in as many blocks as it can while these conditions hold:
- Enough `threads` remain `<- I control by Block/Grid dimensions`
- Enough `warps` remain `<- I control by Block/Grid dimensions`
- Enough `registers` remain `<- I control by writing good code`
- Enough `shared memory` remains?
- Block limit not exceeded?

An example SM might have:
- Max threads per SM: 2048
- Max warps per SM: 64
- Max blocks per SM: 16
- Register file size: say ~256KB
- Shared memory per SM: say ~100KB


## Blocks, grids, and launching
- Blocks:
  - The bread and butter of CUDA
  - Up to a 3-dimensional abstraction of how i want memory sliced up
  - 1D: `dim3 block(256)`
  - 2D: `dim3 block(128, 2)`
  - 3D: `dim3 block(64, 2, 2)`
- Grids:
  - Simply the number of blocks as defined above I'll need to complete my problem space.
  - Undershoot - No error, my problem will just be incomplete
  - Overshoot:
    - Without guards - This will lead to undefined behaviour, e.g. segfault
    - With guards - Should exit immediately, but small overhead cost paid

Apparently the standard CUDA idiom is to launch more threads than necessary and guard them.
The guards:
- Grids:
  - Just ceiling division on the number of blocks needed
    - `ceil(n/x) = (n+x-1)/x`
- Blocks:
  - Always wear protection code wise:
  - Each call of the `vector_add` kernel in a warp knows what its given block and thread idx is. We simply do a calculation that says "if your block index happens to have gone over the maximum which we're also giving you ---n--- then don't bother. You're done"

## Misc Learning
- `__global__` marks a function as a kernel
  - A function that runs of the GPU, but is launched from the host CPU
  - When marked as a kernel, it must therefore necessarily be launched with the `<<<numBlocks, threadsPerBlock>>>` syntax.
  - Also it must return `void`
  - For my include file, `template <typename T>` makes this work for any integer type. Neat