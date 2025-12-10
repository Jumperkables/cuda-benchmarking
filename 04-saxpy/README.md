# 04 - SAXPY
Sum of `ax + y` kernel written here.

### Goal:
- Write it well, getting familiar with CUDA syntax and C++ code
- Benchmark it properly using my new mental model and cheat sheet
  - Do this for a sweep of block sizes.
- Write up my insights and thoughts:
  - acknowledge how I practically can or cannot alter shared memory or register usage substantially with this simple kernel

## Benchmarking
- Achieved memory throughput:
  - Is around 93% for all block sizes, except the largest `104` is at 91%
  - Compute throughput is like 15% for each. Saxpy is very memory bound
  
- Occupancy:
  - `Block Size 64 & 1024`:
    - Occupancy is lower here because theoretical active warps per SM is hamstrung from max 48 down to 32 each.
      - `64` because we hit the max blocks per SM count without maxxing out threads
      - `1024` because max threads per SM is `1536` and we can't fit another block in at this size, leaving 1/3 of the threads on the table.
  - Best occupancy was actually `256`: by a solid 10% (80% vs 70 or worse)
    - `ncu` says because of warp scheduling overheads and imbalanced workload.
      - I think this is because we're simply memory bound

### Best results: Block size 256
Heres a more in-depth profile of my best blok size
- `ncu --section "MemoryWorkloadAnalysis" ./saxpy`
```
==PROF== Disconnected from process 38571
[38571] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (1048576, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Memory Workload Analysis
    --------------------------------------- ----------- ------------
    Metric Name                             Metric Unit Metric Value
    --------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                 0
    Local Memory Spilling Request Overhead            %            0
    L2 Sector Promotion Misses                        %            0
    Shared Memory Spilling Requests                                0
    Shared Memory Spilling Request Overhead           %            0
    Memory Throughput                           Gbyte/s       846.41
    Mem Busy                                          %        39.94
    Max Bandwidth                                     %        92.88
    L1/TEX Hit Rate                                   %            0
    L2 Persisting Size                            Mbyte         1.18
    L2 Compression Success Rate                       %            0
    L2 Compression Ratio                                           0
    L2 Compression Input Sectors                 sector            0
    L2 Hit Rate                                       %        33.13
    Mem Pipes Busy                                    %        15.25
    --------------------------------------- ----------- ------------
```
- `sudo ncu --set full --export roofline ./saxpy`
  - Then inspect it with `ncu-ui roofline.ncu-rep`
  - Go to the "details" tab
  - Check the many sections here

- `ncu --set full ./saxpy`
```
==PROF== Disconnected from process 37944
[37944] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (1048576, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,312,074
    Memory Throughput                 %        93.23
    DRAM Throughput                   %        93.23
    Duration                         ms         3.81
    L1/TEX Cache Throughput           %        19.74
    L2 Cache Throughput               %        40.09
    SM Active Cycles              cycle 5,338,311.68
    Compute (SM) Throughput           %        15.26
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: GPU Speed Of Light Roofline Chart (Overview)
    INF   The ratio of peak float (FP32) to double (FP64) performance on this device is 64:1. The workload achieved
          close to 0% of this device's FP32 peak performance and 0% of its FP64 peak performance. See the Profiling
          Guide (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline) for more details on
          roofline analysis.                                                             

    Section: PM Sampling
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Maximum Buffer Size             Mbyte        26.41
    Maximum Sampling Interval          us            2
    # Pass Groups                                    3
    ------------------------- ----------- ------------

    Section: Compute Workload Analysis
    -------------------- ----------- ------------
    Metric Name          Metric Unit Metric Value
    -------------------- ----------- ------------
    Executed Ipc Active   inst/cycle         0.31
    Executed Ipc Elapsed  inst/cycle         0.31
    Issue Slots Busy               %         7.63
    Issued Ipc Active     inst/cycle         0.31
    SM Busy                        %         7.63
    -------------------- ----------- ------------

    OPT   Est. Local Speedup: 96.19%                                                     
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.

    Section: Memory Workload Analysis
    --------------------------------------- ----------- ------------
    Metric Name                             Metric Unit Metric Value
    --------------------------------------- ----------- ------------
    Local Memory Spilling Requests                                 0
    Local Memory Spilling Request Overhead            %            0
    L2 Sector Promotion Misses                        %            0
    Shared Memory Spilling Requests                                0
    Shared Memory Spilling Request Overhead           %            0
    Memory Throughput                           Gbyte/s       849.59
    Mem Busy                                          %        40.09
    Max Bandwidth                                     %        93.23
    L1/TEX Hit Rate                                   %            0
    L2 Persisting Size                            Mbyte         1.18
    L2 Compression Success Rate                       %            0
    L2 Compression Ratio                                           0
    L2 Compression Input Sectors                 sector            0
    L2 Hit Rate                                       %        33.19
    Mem Pipes Busy                                    %        15.26
    --------------------------------------- ----------- ------------

    Section: Scheduler Statistics
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    One or More Eligible                   %         7.68
    Issued Warp Per Scheduler                        0.08
    No Eligible                            %        92.32
    Active Warps Per Scheduler          warp         9.30
    Eligible Warps Per Scheduler        warp         0.09
    ---------------------------- ----------- ------------

    OPT   Est. Local Speedup: 6.768%                                                     
          Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only
          issues an instruction every 13.0 cycles. This might leave hardware resources underutilized and may lead to
          less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average
          of 9.30 active warps per scheduler, but only an average of 0.09 warps were eligible per cycle. Eligible
          warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no
          eligible warp results in no instruction being issued and the issue slot remains unused. To increase the
          number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons
          on the Warp State Statistics and Source Counters sections.                     

    Section: Warp State Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Warp Cycles Per Issued Instruction             cycle       121.04
    Warp Cycles Per Executed Instruction           cycle       121.06
    Avg. Active Threads Per Warp                                   32
    Avg. Not Predicated Off Threads Per Warp                       30
    ---------------------------------------- ----------- ------------

    OPT   Est. Speedup: 6.768%                                                           
          On average, each warp of this workload spends 113.3 cycles being stalled waiting for a scoreboard dependency
          on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited
          upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the
          memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by
          increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently
          used data to shared memory. This stall type represents about 93.6% of the total average of 121.0 cycles
          between issuing two instructions.                                              
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on
          sampling data. The Profiling Guide                                             
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details
          on each stall reason.                                                          

    Section: Instruction Statistics
    ---------------------------------------- ----------- ------------
    Metric Name                              Metric Unit Metric Value
    ---------------------------------------- ----------- ------------
    Local Memory Spilling Requests                  byte            0
    Shared Memory Spilling Requests                 byte            0
    Avg. Executed Instructions Per Scheduler        inst   409,200.39
    Executed Instructions                           inst  134,217,728
    Avg. Issued Instructions Per Scheduler          inst   409,264.84
    Issued Instructions                             inst  134,238,866
    ---------------------------------------- ----------- ------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                              1,048,576
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,456
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            2,131.25
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
    Achieved Occupancy                        %        76.54
    Achieved Active Warps Per SM           warp        36.74
    ------------------------------- ----------- ------------

    OPT   Est. Speedup: 6.768%                                                           
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (76.5%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,701,144
    Total DRAM Elapsed Cycles        cycle   433,770,496
    Average L1 Active Cycles         cycle  5,338,311.68
    Total L1 Elapsed Cycles          cycle   439,779,546
    Average L2 Active Cycles         cycle  5,285,226.90
    Total L2 Elapsed Cycles          cycle   252,246,288
    Average SM Active Cycles         cycle  5,338,311.68
    Total SM Elapsed Cycles          cycle   439,779,546
    Average SMSP Active Cycles       cycle  5,327,056.74
    Total SMSP Elapsed Cycles        cycle 1,759,118,184
    -------------------------- ----------- -------------

    Section: Source Counters
    ------------------------- ----------- ------------
    Metric Name               Metric Unit Metric Value
    ------------------------- ----------- ------------
    Branch Instructions Ratio           %         0.12
    Branch Instructions              inst   16,777,216
    Branch Efficiency                   %            0
    Avg. Divergent Branches      branches            0
    ------------------------- ----------- ------------
```


### Full Results
#### Block size = 64
```
==PROF== Disconnected from process 35110
[35110] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (4194304, 1, 1)x(64, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,310,824
    Memory Throughput                 %        93.32
    DRAM Throughput                   %        93.32
    Duration                         ms         3.81
    L1/TEX Cache Throughput           %        19.84
    L2 Cache Throughput               %        40.12
    SM Active Cycles              cycle 5,273,281.98
    Compute (SM) Throughput           %        15.46
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    64
    Function Cache Configuration                     CachePreferNone
    Grid Size                                              4,194,304
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,456
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            3,196.88
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           64
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           24
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        53.44
    Achieved Active Warps Per SM           warp        25.65
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 19.85%                                                     
          The difference between calculated theoretical (66.7%) and measured achieved occupancy (53.4%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 33.33%                                                     
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
          hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of blocks that
          can fit on the SM, and the required amount of shared memory.                   

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,725,616
    Total DRAM Elapsed Cycles        cycle   433,661,952
    Average L1 Active Cycles         cycle  5,273,281.98
    Total L1 Elapsed Cycles          cycle   434,050,340
    Average L2 Active Cycles         cycle  5,233,131.54
    Total L2 Elapsed Cycles          cycle   252,181,248
    Average SM Active Cycles         cycle  5,273,281.98
    Total SM Elapsed Cycles          cycle   434,050,340
    Average SMSP Active Cycles       cycle  5,318,547.77
    Total SMSP Elapsed Cycles        cycle 1,736,201,360
    -------------------------- ----------- -------------
```

#### Block size 96
```
==PROF== Disconnected from process 35288
[35288] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (2796203, 1, 1)x(96, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,265,132
    Memory Throughput                 %        93.79
    DRAM Throughput                   %        93.79
    Duration                         ms         3.77
    L1/TEX Cache Throughput           %        20.84
    L2 Cache Throughput               %        40.50
    SM Active Cycles              cycle 5,325,515.18
    Compute (SM) Throughput           %        15.31
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    96
    Function Cache Configuration                     CachePreferNone
    Grid Size                                              2,796,203
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,488
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            2,131.25
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           42
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           16
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        70.68
    Achieved Active Warps Per SM           warp        33.93
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 29.32%                                                     
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (70.7%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,603,012
    Total DRAM Elapsed Cycles        cycle   429,927,424
    Average L1 Active Cycles         cycle  5,325,515.18
    Total L1 Elapsed Cycles          cycle   438,466,948
    Average L2 Active Cycles         cycle  5,196,156.33
    Total L2 Elapsed Cycles          cycle   250,011,120
    Average SM Active Cycles         cycle  5,325,515.18
    Total SM Elapsed Cycles          cycle   438,466,948
    Average SMSP Active Cycles       cycle  5,300,715.90
    Total SMSP Elapsed Cycles        cycle 1,753,867,792
    -------------------------- ----------- -------------
```


#### Block size 256
```
==PROF== Disconnected from process 35456
[35456] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (1048576, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,355,574
    Memory Throughput                 %        92.67
    DRAM Throughput                   %        92.67
    Duration                         ms         3.84
    L1/TEX Cache Throughput           %        19.67
    L2 Cache Throughput               %        39.69
    SM Active Cycles              cycle 5,312,292.28
    Compute (SM) Throughput           %        15.20
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
    Grid Size                                              1,048,576
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,456
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            2,131.25
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
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,773,596
    Total DRAM Elapsed Cycles        cycle   437,324,800
    Average L1 Active Cycles         cycle  5,312,292.28
    Total L1 Elapsed Cycles          cycle   441,533,602
    Average L2 Active Cycles         cycle  5,251,857.08
    Total L2 Elapsed Cycles          cycle   254,312,448
    Average SM Active Cycles         cycle  5,312,292.28
    Total SM Elapsed Cycles          cycle   441,533,602
    Average SMSP Active Cycles       cycle  5,346,433.10
    Total SMSP Elapsed Cycles        cycle 1,766,134,408
    -------------------------- ----------- -------------
```


#### Block size 512
```
==PROF== Disconnected from process 35610
[35610] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (524288, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,283,783
    Memory Throughput                 %        93.56
    DRAM Throughput                   %        93.56
    Duration                         ms         3.79
    L1/TEX Cache Throughput           %        19.83
    L2 Cache Throughput               %        40.23
    SM Active Cycles              cycle 5,285,139.20
    Compute (SM) Throughput           %        15.29
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   512
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                524,288
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,456
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            2,131.25
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            8
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            3
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        67.70
    Achieved Active Warps Per SM           warp        32.50
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 32.3%                                                      
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (67.7%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,639,032
    Total DRAM Elapsed Cycles        cycle   431,461,376
    Average L1 Active Cycles         cycle  5,285,139.20
    Total L1 Elapsed Cycles          cycle   438,983,004
    Average L2 Active Cycles         cycle  5,254,710.81
    Total L2 Elapsed Cycles          cycle   250,902,432
    Average SM Active Cycles         cycle  5,285,139.20
    Total SM Elapsed Cycles          cycle   438,983,004
    Average SMSP Active Cycles       cycle  5,224,834.59
    Total SMSP Elapsed Cycles        cycle 1,755,932,016
    -------------------------- ----------- -------------
```


#### Block size 1024
```
==PROF== Disconnected from process 35761
[35761] saxpy@127.0.0.1
  saxpy(float, const float *, const float *, float *, int) (262144, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle    5,448,948
    Memory Throughput                 %        91.09
    DRAM Throughput                   %        91.09
    Duration                         ms         3.91
    L1/TEX Cache Throughput           %        19.07
    L2 Cache Throughput               %        39.16
    SM Active Cycles              cycle 4,907,495.35
    Compute (SM) Throughput           %        15.03
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing DRAM in the Memory Workload Analysis section.               

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                 1,024
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                262,144
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              82
    Stack Size                                                 1,024
    Threads                                   thread     268,435,456
    # TPCs                                                        41
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                            3,196.88
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            1
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %        66.67
    Achieved Occupancy                        %        50.59
    Achieved Active Warps Per SM           warp        24.29
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 24.11%                                                     
          The difference between calculated theoretical (66.7%) and measured achieved occupancy (50.6%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 33.33%                                                     
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
          hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of warps within
          each block.                                                                    

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle    33,774,848
    Total DRAM Elapsed Cycles        cycle   444,942,336
    Average L1 Active Cycles         cycle  4,907,495.35
    Total L1 Elapsed Cycles          cycle   446,492,110
    Average L2 Active Cycles         cycle  5,386,641.85
    Total L2 Elapsed Cycles          cycle   258,741,072
    Average SM Active Cycles         cycle  4,907,495.35
    Total SM Elapsed Cycles          cycle   446,492,110
    Average SMSP Active Cycles       cycle  4,615,732.88
    Total SMSP Elapsed Cycles        cycle 1,785,968,440
    -------------------------- ----------- -------------
```