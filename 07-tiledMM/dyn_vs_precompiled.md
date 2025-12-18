# Precompiled Tiled Matrix Multiplication vs Dynamic Fallback
Specific analysis on what speedups were achieve and why over dynamic fallback when using a variation that can who arguments are compile time.



## Comparison
- I've pre-compiled a version with:
  - `block.x=32 , block.y=16 , TK=32`
- So a fair comparison will be the dynamic fallback with `block.x and block.y` reversed
  - `block.x=16 , block.x=32 , TK=32`

- Precompiled - Tiling `block.x=32 , block.y=16 , TK=32`
  - `Duration: 18.63ms`
  - `DRAM Throughput: 18.85%`
  - `Peak VRAM usage: 262MB`
- Dynamic Fallback - Tiling `block.x=16 , block.x=32 , TK=32`
  - `Duration: 30.57ms`
  - `DRAM Throughput: 11.59%`
  - `Peak VRAM usage: 358MB`
- Naive Matmul - **NO Tiling** ``
  - `Duration: 27.02ms`
  - `DRAM Throughput: 25.06%`
  - `Peak VRAM usage: 364MB`

- Peak VRAM usage had to be calculated using `nvidia-smi` inspection because `ncu` doesn't monitor raw memory usage
- All the `compute SM throughput` is `95%+` for the above
- The tiling algorithm is noticeably faster
- WHY is it faster?...

### Speedups
1. Static shared memory at compile time
   - When we can assume known size of shared memory in a kernel, we are allowed to validly call static memory with all its advantages:
     - `__shared__ float As[16][16];`
       - This help indexing, but also gives a static and fixed memory block per CUDA block that can be assumed by the compiler
   - `extern` keyword
     - If it ISNT used, the compiler will attempt to store a fixed amount of memory here, which would be wrong
     - `extern ` must be used when relying on dynamic shared memory to tell the compiler that it must essentially be handled dynamically.
   - `__syncthreads()`
     - Obviously this is a halt and wait for all threads
     - This does it for a whole **BLOCK**
     - I'd use `__syncwarp()` for waiting for all threads in a warp instead
     - As to the cost of `__syncthreads()`
       - It isn't inherently expensive
       - The cost is waiting for the slowest warp in the block
   - `#pragma unroll`
     - Brings me back to my undergrad C++ days
     - It replaces a for loop with the same code execution just 16 times in a row
       - So of course its a compile time only constraint, and its a tool i need  to learn to wield

2. More on `__syncthreads()`
   - Think about `divergence`:
     - Where a thread in a warp does something else that might be more expensive
     - `if (tx < TK) load()`
   - `Imbalanced loads`:
```cpp
for (int kk=tx; kk<TK; kk+=BN){
  load();
}
// If BN = 32, but TK = 48...
//... then the first 0-15 threads do 2 iterations, and the others do 1
// Telltale signs of this is `stalled_barriers`
```

### Compile Time Speedups


## Performance Dumps
### Precompiled:
```
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    25,994,943
    Memory Throughput                 %         97.89
    DRAM Throughput                   %         18.85
    Duration                         ms         18.63
    L1/TEX Cache Throughput           %         98.04
    L2 Cache Throughput               %         15.43
    SM Active Cycles              cycle 25,958,206.27
    Compute (SM) Throughput           %         97.89
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
    Registers Per Thread             register/thread              38
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block       Kbyte/block            6.14
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
    Achieved Occupancy                        %        99.58
    Achieved Active Warps Per SM           warp        47.80
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- -------------
    Metric Name                Metric Unit  Metric Value
    -------------------------- ----------- -------------
    Average DRAM Active Cycles       cycle 33,351,409.33
    Total DRAM Elapsed Cycles        cycle 2,122,648,576
    Average L1 Active Cycles         cycle 25,958,206.27
    Total L1 Elapsed Cycles          cycle 2,131,757,126
    Average L2 Active Cycles         cycle 25,542,931.17
    Total L2 Elapsed Cycles          cycle 1,234,346,544
    Average SM Active Cycles         cycle 25,958,206.27
    Total SM Elapsed Cycles          cycle 2,131,757,126
    Average SMSP Active Cycles       cycle 25,958,913.59
    Total SMSP Elapsed Cycles        cycle 8,527,028,504
    -------------------------- ----------- -------------
```



### Dynamic Fallback:
```
  void tiledMM_naive_dyn<float>(const T1 *, const T1 *, T1 *, int, int, int, int) (188, 125, 1)x(16, 32, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          9.49
    SM Frequency                    Ghz          1.39
    Elapsed Cycles                cycle    42,667,572
    Memory Throughput                 %         94.82
    DRAM Throughput                   %         11.59
    Duration                         ms         30.59
    L1/TEX Cache Throughput           %         94.97
    L2 Cache Throughput               %         14.20
    SM Active Cycles              cycle 42,600,356.48
    Compute (SM) Throughput           %         94.82
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
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            6.14
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
    Achieved Occupancy                        %        99.64
    Achieved Active Warps Per SM           warp        47.83
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- --------------
    Metric Name                Metric Unit   Metric Value
    -------------------------- ----------- --------------
    Average DRAM Active Cycles       cycle  33,644,421.33
    Total DRAM Elapsed Cycles        cycle  3,484,079,104
    Average L1 Active Cycles         cycle  42,600,356.48
    Total L1 Elapsed Cycles          cycle  3,498,862,250
    Average L2 Active Cycles         cycle  41,846,346.88
    Total L2 Elapsed Cycles          cycle  2,026,033,488
    Average SM Active Cycles         cycle  42,600,356.48
    Total SM Elapsed Cycles          cycle  3,498,862,250
    Average SMSP Active Cycles       cycle  42,608,323.35
    Total SMSP Elapsed Cycles        cycle 13,995,449,000
    -------------------------- ----------- --------------
```