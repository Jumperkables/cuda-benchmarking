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
