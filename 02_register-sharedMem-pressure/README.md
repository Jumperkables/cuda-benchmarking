# 02 - Register and Shared Memory Pressure
- In `01_vector-add` I've already learned how to mess with block size and grid size to change occupancy
- Here, I'm going to intentionally mess with the code here to get used to how it changes performance
- Then next increasingly complicated lessons looking at stuff like coalescence I'll be able to monitor these things and figure out occupancy

## Misc learning
- `const` and `__restrict__` keywords in C++
  - `const` is a promise to the compiler and also a stylistic hint that the value won't be changing during this function call. Doesn't mean its globally immutable of course
  - `__restrict__` is a keyword that guarantees arrays will not overlap memory, allowing the compiler to assume changes to one do not affect the others

## General details on Occupancy:
CUDA's [best practice guide section on occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy) is quite useful for this. Notable points I should remember:
- Register allocations are rounded up to the nearest 256 registers per warp
- For CUDA Compute 7.0, each SM has 65,536 32 bit registers
  - `65,536 = (a)32 * (b)64 * (c)32`
    - `(a)` = `32` threads in a warp
    - `(b)` = `64` warps that can be fit on any SM at one time
    - `(c)` = `32` At most 32 registers per thread if you want maximum occupancy. Going above will cause rounding upwards.

## Register size
Lets sart with playing with register size.
- Maximising occupancy by minimising register size i.e. don't write wasteful code and you'll maximise occupancy by registers
- `--ptxas-options=v` option for `nvcc` gives the number of registers used per thread
- There is also an ~~[occupancy calculator](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator) from Nsight Compute which I'll play around with now.~~
  - Turns out this doesn't exist right now. Its down on their page
  - Use this [xmartlabs calculator](https://xmartlabs.github.io/cuda-calculator/) instead.

### 1) Standard vec_add (as in part 01)
- Original vec_add is already pretty well optimised
- code
```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];   // Not very wasteful at all
    }
}
```
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z10vector_addPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z10vector_addPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.168 ms
```


### 2) Being wasteful with registers:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= n) return;

float ax = a[idx];
float bx = b[idx];

// Purposeful bloat
float t0 = ax * 1.111f;
float t1 = t0 * 1.111f;
float t2 = t1 * 1.111f;
float t3 = t2 * 1.111f;
float t4 = t3 * 1.111f;
float t5 = t4 * 1.111f;

float u0 = bx * 0.999f;
float u1 = u0 * 0.999f;
float u2 = u1 * 0.999f;
float u3 = u2 * 0.999f;
float u4 = u3 * 0.999f;
float u5 = u4 * 0.999f;

float r = t5 + u5;
c[idx] = r;
```
- This first above attempt didn't actually work, register usage was the same:
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20vector_add_reg_heavyPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z20vector_add_reg_heavyPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.285 ms
```
- Register usage was the same because the compiler figured out it could fold the above chains into something that uses just 1 register:
```cpp
ax = ax * 1.1f;
ax = ax * 1.1f;
ax = ax * 1.1f;
ax = ax * 1.1f;
ax = ax * 1.1f;
ax = ax * 1.1f;
```
- making it harder
```cpp
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ax = a[idx];
    float bx = b[idx];

    float t0 = ax * 1.1f;
    float t1 = bx * 0.9f;
    float t2 = t0 + t1;

    float t3 = t2 * 1.1f;   // keep t2 alive
    float t4 = t3 + t1;     // keep t1 alive
    float t5 = t4 + t0;     // keep t0 alive
    float t6 = t5 + t3;     // keep t3 alive
    float t7 = t6 + t4;     // keep t4 alive
    float t8 = t7 + ax;     // keep ax alive
    float t9 = t8 + bx;     // keep bx alive
    float result = t9;

    c[idx] = result;
```
- The above only increase the register count from `12 -> 14`
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20vector_add_reg_heavyPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z20vector_add_reg_heavyPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 1.285 ms
```


## 3) Really pushing the number of registers further
- ChatGPT tells me this code will push registers up to the limit:
```cpp
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float v0 = ax * (i + 1);
        float v1 = bx * (i + 2);
        acc0 += v0;
        acc1 += v1;
        acc2 += v0 * 1.1f;
        acc3 += v1 * 1.2f;
        acc4 += v0 * 1.3f;
        acc5 += v1 * 1.4f;
        acc6 += v0 * 1.5f;
        acc7 += v1 * 1.6f;
    }

    float sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    c[idx] = sum;
```
- Okay it didn't push the number of registers as far as i thought it might:
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20vector_add_reg_heavyPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z20vector_add_reg_heavyPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 4.540 ms
```
- Only 22 now. But now I can introduce myself to a concept called spilling
- Scratch that, I can only do that if I get over the lower bound of 24

```cpp
    float ax = a[idx];
    float bx = b[idx];

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    float acc4 = 0.f, acc5 = 0.f, acc6 = 0.f, acc7 = 0.f;
    float acc8 = 0.f, acc9 = 0.f, acc10 = 0.f, acc11 = 0.f;

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        float v0 = ax * (i + 1);
        float v1 = bx * (i + 2);
        acc0 += v0;
        acc1 += v1;
        acc2 += v0 * 1.1f;
        acc3 += v1 * 1.2f;
        acc4 += v0 * 1.3f;
        acc5 += v1 * 1.4f;
        acc6 += v0 * 1.5f;
        acc7 += v1 * 1.6f;
        acc8 += v0 * 1.7f;
        acc9 += v1 * 1.8f;
        acc10 += v0 * 1.9f;
        acc11 += v1 * 2.0f;
    }

    float sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10 + acc11;
```
- Ok, the above now has 34 registers
```
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20vector_add_reg_heavyPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z20vector_add_reg_heavyPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 34 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 7.009 ms
```
- So now i can try to enforce the max register count and see what compromises the compiler has to make:
  - `nvcc -Xptxas -maxrregcount=24 --ptxas-options=-v vector_add_wasteful-registers.cu -o vector_add_wasteful-registers`
```
ptxas info    : Overriding maximum register limit 256 for '_Z20vector_add_reg_heavyPKfS0_Pfi' with  24 of maxrregcount option
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z20vector_add_reg_heavyPKfS0_Pfi' for 'sm_75'
ptxas info    : Function properties for _Z20vector_add_reg_heavyPKfS0_Pfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 24 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compile time = 6.121 ms
```
- It didn't spill, but it did manage to find its way to get the registers to 24 again. So this will be a cool thing to benchmark. Registers 34 vs registers 24.
- 34 should decrease the theoretical occupancy as the compiler is forced to round up to the nearest for register uses.
#### 3a) 34 registers
```
ister-sharedMem-pressure/vector_add_wasteful-registers_reg34)
==PROF== Profiling "vector_add_reg_heavy" - 0: 0%....50%....100% - 8 passes
14568.000000 14568.000000 14568.000000 14568.000000 14568.000000 ==PROF== Disconnected from process 10968
[10968] vector_add_wasteful-registers_reg34@127.0.0.1
  vector_add_reg_heavy(const float *, const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle      843,759
    Memory Throughput                 %        36.40
    DRAM Throughput                   %        36.40
    Duration                         us       605.76
    L1/TEX Cache Throughput           %         8.51
    L2 Cache Throughput               %        15.72
    SM Active Cycles              cycle   840,217.88
    Compute (SM) Throughput           %        89.59
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 65,536
    Registers Per Thread             register/thread              36
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
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        86.89
    Achieved Active Warps Per SM           warp        41.71
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 13.11%                                                     
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (86.9%) can be the
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on
          optimizing occupancy.                                                          

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle 2,092,954.67
    Total DRAM Elapsed Cycles        cycle   68,995,072
    Average L1 Active Cycles         cycle   840,217.88
    Total L1 Elapsed Cycles          cycle   69,206,272
    Average L2 Active Cycles         cycle   807,983.21
    Total L2 Elapsed Cycles          cycle   40,123,584
    Average SM Active Cycles         cycle   840,217.88
    Total SM Elapsed Cycles          cycle   69,206,272
    Average SMSP Active Cycles       cycle   839,598.66
    Total SMSP Elapsed Cycles        cycle  276,825,088
    -------------------------- ----------- ------------
```

#### 3b) 24 registers (below 32)

```
==PROF== Connected to process 11088 (/home/jumperkables/projects/cuda-benchmarking/02_register-sharedMem-pressure/vector_add_wasteful-registers_reg24)
==PROF== Profiling "vector_add_reg_heavy" - 0: 0%....50%....100% - 8 passes
14568.000000 14568.000000 14568.000000 14568.000000 14568.000000 ==PROF== Disconnected from process 11088
[11088] vector_add_wasteful-registers_reg24@127.0.0.1
  vector_add_reg_heavy(const float *, const float *, float *, int) (65536, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         9.49
    SM Frequency                    Ghz         1.39
    Elapsed Cycles                cycle      837,706
    Memory Throughput                 %        37.50
    DRAM Throughput                   %        37.50
    Duration                         us       602.34
    L1/TEX Cache Throughput           %         8.70
    L2 Cache Throughput               %        15.98
    SM Active Cycles              cycle      832,604
    Compute (SM) Throughput           %        90.51
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of this device.
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.
          Start by analyzing workloads in the Compute Workload Analysis section.         

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   256
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                 65,536
    Registers Per Thread             register/thread              24
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
    Block Limit Registers                 block           10
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        90.87
    Achieved Active Warps Per SM           warp        43.62
    ------------------------------- ----------- ------------

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle    2,144,228
    Total DRAM Elapsed Cycles        cycle   68,606,976
    Average L1 Active Cycles         cycle      832,604
    Total L1 Elapsed Cycles          cycle   68,502,734
    Average L2 Active Cycles         cycle   782,155.52
    Total L2 Elapsed Cycles          cycle   39,896,352
    Average SM Active Cycles         cycle      832,604
    Total SM Elapsed Cycles          cycle   68,502,734
    Average SMSP Active Cycles       cycle      831,982
    Total SMSP Elapsed Cycles        cycle  274,010,936
    -------------------------- ----------- ------------
```

We can see the occupancy difference between the 34 thread and 24 thread versions here:
```
// 34 thread
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            6
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            6
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        86.89
    Achieved Active Warps Per SM           warp        41.71
    ------------------------------- ----------- ------------

// 24 thread
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
    Achieved Occupancy                        %        90.87
    Achieved Active Warps Per SM           warp        43.62
    ------------------------------- ----------- ------------
    
// 24 thread, and block size smaller 256 -> 128
    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           21
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block           12
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        90.81
    Achieved Active Warps Per SM           warp        43.59
    ------------------------------- ----------- ------------
```

## Something Important I learned today:
- Theoretical number of warps utilised is NOT something to optimise directly
- Active warps are a **latency hiding tool**
  - You want enough warps active until the bottleneck is maximally utilised:
    - Memory bottleneck
    - Compute bound ALU
  - After that, more warps are useless or harmful