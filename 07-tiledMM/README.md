# Tiled Matrix Multplication
Now that I understand the memory throughput problems with my naive matrix multiplication kernel, time to implement a tiled MM variation to address them.


# Benchmarking vs Naive
Lets see how this `tiledMM` stacks up against my naive implementation 

## Dynamic Fallback vs Precompiled assigned at runtime
- See a comparison of my findings in this [accompanying README](dyn_vs_precompiled.md), otherwise this file will get too long.

## GFLOP/s Analysis:
- Naive: `2 x M x K x N`
- Tiled: ``

## Tiled Parameter Sweep
- Note that this implementation is the dynamic fallback version, which won't get certain advantages that a hard coded compiled menu kernel will
- For `M,K,N = 4000,2000,3000`

### Block.x|Block.y|TileK = 16|16|64
Worse than naive currently
```
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    42,918,954
    Memory Throughput                 %         98.53
    DRAM Throughput                   %         22.20
    Duration                         ms         30.77
    L1/TEX Cache Throughput           %         98.66
    L2 Cache Throughput               %         18.63
    SM Active Cycles              cycle 42,858,855.99
    Compute (SM) Throughput           %         98.53
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
    Registers Per Thread             register/thread              40
    Shared Memory Configuration Size           Kbyte           65.54
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            8.19
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
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block            7
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        99.62
    Achieved Active Warps Per SM           warp        47.82
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle  64,847,185.33
    Total DRAM Elapsed Cycles        cycle  3,504,709,632
    Average L1 Active Cycles         cycle  42,858,855.99
    Total L1 Elapsed Cycles          cycle  3,519,042,398
    Average L2 Active Cycles         cycle  42,353,452.77
    Total L2 Elapsed Cycles          cycle  2,038,030,560
    Average SM Active Cycles         cycle  42,858,855.99
    Total SM Elapsed Cycles          cycle  3,519,042,398
    Average SMSP Active Cycles       cycle  42,866,251.51
    Total SMSP Elapsed Cycles        cycle 14,076,169,592
    -------------------------- ----------- --------------
```

### Block.x|Block.y|TileK = 16|16|128
- Note that max theoretical warps/SM has reduced to 40:
  - `Block Limit Shared Mem` under occupancy is the root cause:
    - Under `Section: Launch Statistics`
      - `Shared Memory Config Size = 102.4KB` i.e. the max shared memory per SM
      - Driver shared memory per block + dynamic shared memory per block = 
        - 1.02KB + 16.38KB = `17.4KB`
        - If there were 6 blocks, = `17.4*6 = 104.4KB`
        - This is just above the shared mem per SM limit. Hence we must drop to 5 blocks for this level of shared memory from the `tile_k = 128`
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    42,952,334
    Memory Throughput                 %         98.47
    DRAM Throughput                   %         22.18
    Duration                         ms         30.79
    L1/TEX Cache Throughput           %         98.65
    L2 Cache Throughput               %         18.67
    SM Active Cycles              cycle 42,863,038.43
    Compute (SM) Throughput           %         98.47
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
    Registers Per Thread             register/thread              40
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block           16.38
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread      12,032,000
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                              114.63
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           40
    Theoretical Occupancy                     %        83.33
    Achieved Occupancy                        %        83.07
    Achieved Active Warps Per SM           warp        39.87
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle     64,812,824
    Total DRAM Elapsed Cycles        cycle  3,507,345,408
    Average L1 Active Cycles         cycle  42,863,038.43
    Total L1 Elapsed Cycles          cycle  3,521,512,154
    Average L2 Active Cycles         cycle  42,237,210.71
    Total L2 Elapsed Cycles          cycle  2,039,562,480
    Average SM Active Cycles         cycle  42,863,038.43
    Total SM Elapsed Cycles          cycle  3,521,512,154
    Average SMSP Active Cycles       cycle  42,883,736.72
    Total SMSP Elapsed Cycles        cycle 14,086,048,616
    -------------------------- ----------- --------------
```

### Block.x|Block.y|TileK = 32|16|128
- Duration is still worse than my naive one for now
- Found an occupancy with slightly larger block size allowing larger shared memory
- According to ChatGPT, the relative slowdown is likely because of the `__syncthreads()` overhead and some instruction mix.
- I'm getting a little too into the depths of benchmarking right now, i'm going to go ahead and make the other templated ready-made kernel
```
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    41,416,489
    Memory Throughput                 %         99.28
    DRAM Throughput                   %         23.00
    Duration                         ms         29.69
    L1/TEX Cache Throughput           %         99.41
    L2 Cache Throughput               %         14.59
    SM Active Cycles              cycle 41,357,471.99
    Compute (SM) Throughput           %         99.28
    ----------------------- ----------- -------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   512
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 23,500
    Registers Per Thread             register/thread              40
    Shared Memory Configuration Size           Kbyte          102.40
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block           24.58
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
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block            4
    Block Limit Warps                     block            3
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        99.63
    Achieved Active Warps Per SM           warp        47.82
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle  64,820,054.67
    Total DRAM Elapsed Cycles        cycle  3,382,018,048
    Average L1 Active Cycles         cycle  41,357,471.99
    Total L1 Elapsed Cycles          cycle  3,395,621,852
    Average L2 Active Cycles         cycle  40,020,005.17
    Total L2 Elapsed Cycles          cycle  1,966,683,552
    Average SM Active Cycles         cycle  41,357,471.99
    Total SM Elapsed Cycles          cycle  3,395,621,852
    Average SMSP Active Cycles       cycle  41,350,778.02
    Total SMSP Elapsed Cycles        cycle 13,582,487,408
    -------------------------- ----------- --------------
```


## Best Tiled - Dynamic:
- `Block size: ?? x ??`
- Occupancy:
- Duration:
- Memory Bottleneck:
- Compute:

## Best Naive:
- `Block size: 16 x 16`
- Occupancy:
- Duration:
- Memory Bottleneck:
- Compute:
```
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    37,691,260
    Memory Throughput                 %         97.49
    DRAM Throughput                   %         25.07
    Duration                         ms         27.02
    L1/TEX Cache Throughput           %         97.56
    L2 Cache Throughput               %         21.20
    SM Active Cycles              cycle 37,628,585.41
    Compute (SM) Throughput           %         97.49
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
    Average DRAM Active Cycles       cycle  64,306,737.33
    Total DRAM Elapsed Cycles        cycle  3,077,972,992
    Average L1 Active Cycles         cycle  37,628,585.41
    Total L1 Elapsed Cycles          cycle  3,087,865,558
    Average L2 Active Cycles         cycle  37,245,139.92
    Total L2 Elapsed Cycles          cycle  1,789,876,032
    Average SM Active Cycles         cycle  37,628,585.41
    Total SM Elapsed Cycles          cycle  3,087,865,558
    Average SMSP Active Cycles       cycle  37,632,763.10
    Total SMSP Elapsed Cycles        cycle 12,351,462,232
    -------------------------- ----------- --------------
```


# Interesting Learning:
- I learned you can template CUDA kernels
```cpp
template<int BM, int BN, int TK>
__global__ void tiled_MM(...){
    ...
}
```
- This templating here allows assumptions to be made at compilation time, not at run time
- Apparently this allows a bunch of benefits including:
  - Better loop unrolling
  - More optimal memory layout
  - Not needing to use extern for some reason

### Templating, Compile time variations, and Dynamic fallbacks
I've just learned that often in professional level CUDA code, people write dynamic fallbacks, and hard compile variations. So I'm going to start by making a dynamic fallback version. 
- template<typename T> DOESN'T pay by being unknown at compile time, the compiler figures it out
- But it DOES make the binary larger, which can be a problem