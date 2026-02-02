# **CUDA Kernel Design Cheatsheet**

```
=======================================================
Device 0: "NVIDIA GeForce RTX 3090"
=======================================================
Compute Capability:                 8.6
UUID:                               5c57ee1534b0285a574850b4b0975c8c
Streaming Multiprocessors (SMs):    82
Warp Size:                          32
Max Threads / SM:                   1536
Max Threads / Block:                1024
Max Block Dimensions:               1024 x 1024 x 64
Max Grid Dimensions:                2147483647 x 65535 x 65535
Registers per Block:                65536
Registers per SM:                   65536
Shared Memory per Block:            49152 bytes
Shared Memory per SM:               102400 bytes
Total Constant Memory:              65536 bytes
Global Memory:                      25286672384 bytes
Memory Bus Width:                   384 bits
L2 Cache Size:                      6291456 bytes
Concurrent Kernels:                 Yes
ECC Enabled:                        No
Async Engine Count (DMA engines):   2
Unified Addressing:                 Yes
Max Texture 1D Size:                131072
Max Texture 2D Size:                131072 x 65536
Max Texture 3D Size:                16384 x 16384 x 16384

=======================================================
Device 1: "NVIDIA GeForce RTX 5060 Ti"
=======================================================
Compute Capability:                 12.0
UUID:                               077cc6652f3c22978546435c875801c6
Streaming Multiprocessors (SMs):    36
Warp Size:                          32
Max Threads / SM:                   1536
Max Threads / Block:                1024
Max Block Dimensions:               1024 x 1024 x 64
Max Grid Dimensions:                2147483647 x 65535 x 65535
Registers per Block:                65536
Registers per SM:                   65536
Shared Memory per Block:            49152 bytes
Shared Memory per SM:               102400 bytes
Total Constant Memory:              65536 bytes
Global Memory:                      16617897984 bytes
Memory Bus Width:                   128 bits
L2 Cache Size:                      33554432 bytes
Concurrent Kernels:                 Yes
ECC Enabled:                        No
Async Engine Count (DMA engines):   2
Unified Addressing:                 Yes
Max Texture 1D Size:                131072
Max Texture 2D Size:                131072 x 65536
Max Texture 3D Size:                16384 x 16384 x 16384
```

## FYI Typical orders of magnitude
Say with me our daily mantra:
- ON GPUS, MEMORY LATENCY DOMINATES **EVERYTHING**.
- Arithemetic is cheap. 
- Addressing is cheap.
- Waiting is expensive
- If you can recompute something instead of loading it: DO IT
- Load once and re-use many times: DO IT

### Arithmetic
| Operation            | Approx cycles      | Notes                |
| -------------------- | ------------------ | -------------------- |
| FP32 FMA (`a*b + c`) | **1 cycle**        | Fully pipelined      |
| FP32 add or mul      | 1 cycle            | Same                 |
| Integer add          | 1 cycle            |                      |
| Integer mul          | ~4 cycles          | Still cheap          |
| Integer divide       | **very expensive** | Avoid in inner loops |

### Addresses
| Operation                      | Cost                  |
| ------------------------------ | --------------------- |
| Address add                    | 1 cycle               |
| Address multiply (runtime)     | ~4 cycles             |
| Compile-time constant multiply | **0 cycles** (folded) |

Why compile time matters:
ptr + (ty * 16 + k)   // folded, cheap
ptr + (ty * TK + k)  // runtime multiply if TK not known


## Memory
- Note that global loads are done in 128 Byte segments
- Register: ~1 cycle
  - `Private to each thread`
  - Each soldier carries and is given these, no sharing.
- Shared mem: ~20-30 cycles
  - `Shared between block` 
  - Bunker cache brought in by the admin team. The whole block shares this.
- L1/TEX (Texture): 30-50 cycles
  - Warehouse for the 1 port
  - `Shared per SM`

`Shared by ALL SMs:`
- L2: ~100 cycles
  - Base storage center
  - Hardware managed
- DRAM: ~400-800 cycles
  - Back in home country
  - Off chip
  - High latency, high bandwidth, so we want many requests at once here



## Compilation and profiling
### How many registers?
- `nvcc -O3 -lineinfo -Xptxas=-v ...`
  - Shows register spill
- Fixing the max register count:
  - `nvcc -Xptxas -maxrregcount=24 --ptxas-options=-v ...`


## Profiling
- `sudo ncu ./executable`
- `sudo ncu --set full --export roofline ./executable`
  - Then inspect it with `ncu-ui roofline.ncu-rep`
  - Go to the "details" tab
  - Check the many sections here
- `ncu --set full ./executable`


## Reading a new Kernel?
1. Count Bytes:
    - Read/Writes from global memory
    - Read/Writes from shared memory
2. Think about problem algorithmically
    - What bottleneck SHOULD it be?
    - Don't optimise the wrong one lol
3. Count FLOPs:
4. Memory access and Operation Ordering: 
    1. Coalescence (are warp reads contiguous, yes or no?)
    2. Segments
        - 95% of the time, global memory loads are 128B = 32x4
        - Do my loads execute in 1 or more transactions?
    3. Tail
        - Small problem size = how many wasted bytes
    4. Instruction Level Parallelism
        - Instruction chaining e.g. num accumulators
    5. Parallelism: Enough blocks/warps overall to cover estimated load times?


## Misc 
- Spotting compile time speedups:
  - `#pragma unroll`
  - Register count:
    - higher register count implies static kernels = good sign
  - Higher instruction count for similar algorithm
  - FLOP/s per instruction increase
- Not using excessive generality
  - Good python is often general for re-usability
  - Good CUDA will not let excessive generality cause things unnecessary slowdown such as introducing redundent compile-time ambiguity


## Rules of thumb
### **1. Register Usage**

**Hardware limits (per SM):**

* **65,536 registers**
* **1536 threads**
* **48 warps**
* **Register allocation granularity = 8 registers/thread**

**Guidelines:**

* Aim for **≤ 40 registers per thread**
  (`65536 regs / 1536 threads ≈ 42.6`, rounded down to the nearest 8 = **40**)
* Allow higher if reducing registers causes **spills** (benchmark both versions).
* Monitor impact on:
  * number of resident blocks
  * number of resident warps
  * achieved occupancy

---

### **2. Block Size Selection**

**Hardware limits:**

* Max threads per block: **1024**
* Max threads per SM: **1536**
* Max blocks per SM: **16**
* Warp size: **32**

**Guidelines:**

* Block size must be a **multiple of 32**.
* Avoid very small blocks (< **96** threads).
  (`1536 / 16 = 96` → smaller wastes SM block slots.)
* Prefer block sizes that **divide 1536 cleanly**:

  * **128, 192, 256, 384, 512** (all map nicely to 48 warps)
* Be cautious with **1024-thread blocks**:

  * Only 1 block/SM → 32 warps → lower theoretical occupancy.
* Powers of 2 are **not required** by hardware.

  * But convenient for algorithms with halve-and-shift reduction patterns.

---
### **3. Occupancy and Warp Count**

**Hardware limits (per SM):**

* Max threads: **1536**
* Max warps: **48**

Occupancy goals:

* ≥ **16–24 warps** usually hide latency well.
* ≥ **32 warps** is often indistinguishable from full occupancy.
* Full occupancy (48/48) is **not required** for peak performance.

Focus on:
* having “enough” warps,
* not chasing maximum warps.

---

# **4. Shared Memory Usage**

**Hardware shared memory limits:**

* **Static** Shared memory per block: **49,152 bytes** (48 KB)
  * There can be even more dynamic shmem allowed, and this all also depends on GPU architecture. See Hopper
* Shared memory per SM: **102,400 bytes** (~100 KB)

**Effects:**

* Shared memory is allocated **per block**.
* Blocks with high shared-memory use reduce:

  * blocks per SM
  * warps per SM
  * theoretical occupancy

**Guidelines:**

* Allocate only what is needed.
* Prefer patterns that heavily **reuse** SMEM.
* Profile SMEM pressure in Nsight Compute (Launch Stats → SMEM usage).

---

# **5. Algorithmic Behavior — TODO**

I will fill this section after learning:

* Memory coalescing
* Memory hierarchy behavior (L1, L2, cache thrashing)
* Shared-memory bank conflicts
* Global→shared tiling
* Divergence reduction strategies
* Warp-level primitives (`__shfl_sync`, `__ballot_sync`)
* Reconvergence model

This section will eventually become one of the most important.

---

# **6. Occupancy Isn’t Everything**

Once you have enough resident warps to hide latency:

* Focus shifts to reducing total **instructions**.
* Improve **memory efficiency** (coalescing, tiling, caching).
* Minimize **branch divergence**.
* Minimize **redundant loads/stores**.
* Improve **arithmetic intensity**.

These optimizations often yield more benefit than increasing occupancy.

---

# **7. Profiling Checklist (Nsight Compute)**

Always verify:

* Register usage
* Shared memory usage
* Blocks per SM
* Warps per SM (theoretical + achieved)
* Warp stall reasons
* L1/L2 hit rates
* Memory throughput %
* ALU utilization %
* Divergence metrics
* DRAM read/write footprint

Use measurements to guide kernel design, not assumptions.

---

## Analogy
### Updates I need to make to the analogy:
- [2025-12-12]: While doing matmul i discovered that you can have underfilled warps, if you define a block size of 1.
  - This only launches 1 THREAD per block, which goes to 32 lanes (warp)
  - My analogy needs to define closely that the 32 lanes are the warp, and that sometimes less soldiers can be deployed than 32

### [2025-12-10] Updated Mental Model - Military Operation
- Yesterday's kitchen analogy was slightly flawed, because I thought of each of the threads in the warp as units of work, as opposed to the workers themselves. It would be more apt to deliver workers than burgers.
- So, an improved analogy might be a military operation:

- Each `SM` is a dock
- Each `block` of work is a ship containing soldiers
- There are only 16 ports per dock
  - `16 blocks / SM`
- Each `warp` is a squad of 32 soldiers
  - Squads are always size 32
- Each `thread` inside a warp is a soldier
- Each `SM` has 4 `warp schedulers`
  - These are 4 drill seargants
- Each `block` can be not larger than `2048`
  - Only max 2048 soldiers can fit on each ship
- The `kernel` is preparation each soldier must do.
  - Complexity of the kernel is analogous to the amount of preparation resources each squadron requires
- `Shared memory` is shared between every `warp` squadron in a given `block` ship.
  - Each `SM` has a budget of `100KB` of shared memory.
  - Interesting, each `block` can ask for up to `50KB` of shared memory each.
- `Shared memory` is the amount of ship staff required to handle logistics for the the entire block squadrons preparations.
  - Each of the 16 ports in a single dock, each block, must share logistics capacity.
  - Each single dock only has enough admin capacity for 100 logistics workers.
  - However, each ship can come in with up to 50 logistics workers each.
  - Since the logistics workers must be cleared for landing before they can dock, if the misson happens to require a lot of `shared memory` logistics, then it doesn't matter if only 2 `block` ships at a time can land, with the other 14 `active resident block` ports left empty, the other ships are NOT docking
  - Its up to the `coder` military command to design a better mission `kernel` if he wants to use the full `resident` capacity of the docks by being less wasteful with `shared memory` logistics.
- Each `SM` has a `maximum theoretical activate warp` capacity dictated indirectly by `max threads / SM`.
  - On both my GPUs its `1536 = 48 x 32 = 48 theoretical max warps`
  - I hear other GPUs have `2048 = 64 x 32 = 64 theoretical max warps`
  - Each dock only has capacity for `1536` soldiers to do preparations.
- `Registers` are the number of tools each soldier requires for their preparation.
  - Each entire dock comes equipped with enough `registered` tools for somewhere in the region of `~40-60 per thread` soldier. Mine are `65536 registers per SM`.
  - Interestingly, a single `block` is allowed to ask for ALL `65536 registers`
    - For me, each `thread` soldier can have `65536/1536 registered = 42.6` tools before 
    - Assuming everything else is maximally efficient, if the job requires more than `42 
    - Note that registers are allocated per thread in multiples of something like 8. In practice, If i need more than 40, I'll bump right up to 48.
  registered` tools per soldier, then less `block` ships will be allowed to land for `resident active` duty as there isn't the tool for them, leaving some stations unattended.
- The `SM` has `4 warp scheduling` drill sergeants:
  - Each `active warp` from a `resident block` is given orders by the drill sergeants. Who wander around very quickly keeping themselves occupied.
  - They give individual orders to warp squadrons:
    - "You `32 threads in warp 12`, go wait in that line you share with the other `warps` to fetch your `memory` uniform"
    - "You `32 threads in warp 47`, report to `Float 32 ALU` with your `memory uniform` at once!
  - If he checks on a `warp` that is busy carrying out his orders, he moves to check on another `warp` squad.
- Think carefully now, the true bottleneck here is how long his orders take:
  - How quickly the `memory fetch` line moves
  - How quickly the `ALU` stations can service each `warp` squad.
- Have too few `warp` squads in the docks, the `ALU and memory` stations are underused and idling.
- Have too many `warp` squads, and it won't make things any faster. The `ALU and memory` lines are already working at peak. It does run the risk of accidentally overusing `shared memory` or `registers`.

My job:
- Write instruction `kernels` that balance the `ALU and memory` demands efficiently, preferably maxxing both out.
- Order in just enough `warps` to `hide the latency` keeping one or both `ALU and memory` as busy as possible.
  - Ordering in more `warps` might not harm things, but could run the risk of hogging too many `registers` or `shared mem` and exacerbating the common pitfalls below. 
- Aim for `40 or less registers per thread` - `65536 max registers per SM / 1536 max threads per SM <-` rounded down to the nearest 8.
  - Allow more if spilling would happen and cause slowdown, just benchmark it versus reduced thread and warp usage from register limit.
- Aim for `block size greater than 96` - `1536 max threads per SM / 16 max blocks per SM`
- Aim for `block size` that divides `1536 max threads per SM cleanly`:
  - Dont go above `768` unless you've a bloody good reason. You'll have to cap out at `1024` leave over 500 threads on the table.
- DONT be constrained by block size of powers of 2: `128, 256, 512`
  - Simply make sure its a `multiple of 32`
  - Though NOTE some algorithm i'll be writing kernels for work better and even assume powers of 2:
    - Bit tricks, masking, and shifting: `for (int s = blockDim.x / 2; s > 0; s >>= 1)`
- **High occupancy is not everything**. Once latency is covered, algorithmic things matter more like:
  - Coalesing
  - Fewer instructions
  - Reduced divergence

Common pitfalls:
- Underusing the `ALU and memory` because:
  - Too few `blocks` are allowed to dock:
    - Each `block` is taking too much `shared memory` logistics. Can easily accidentally let only 2 `blocks` at a time in.
      - `Max Shared mem per SM: 102KB`
      - `Max shared mem per block: 49KB`
    - Each `block` is taking all the `registered tools` the dock can supply. Giving them simpler `kernel` tasks would alleviate this.
      - `Max registers per SM: 65536`
      - `Max registers per block: 65536`
    - The `block` boat size is too large, this is VERY easy to do:
      - `Max threads per SM: 1536 = 48 warps max`
      - `Max threads per Block: 1024 = 32 warps max`
      - A naively large block size would leave space for only `1 block` 
    - The `block` boat size is too small, this is VERY easy to do:
      - `1536 max threads per SM / 16 max blocks per SM = 96 threads per block`
      - If i pick anything under `96 threads per block`, I'm leaving threads on the table

### Advanced Analogy:
- Latency hiding is kind of a thing only because ALUs are shared. If every warp had its own ALU, then WARP B wouldn't be able to hide its latency while it waits for memory. Also instruction fetch, and memory pipelines.
- Each cycle in different memory movement is made of essential steps that are all quite fast in themselves. i.e. a 400 cycle DRAM read is NOT just 399 cycles of prep and 1 cycle of movement, its more like very few steps each quite expensive due to off-chip travel, queuing, arbitration,and DRAM row activations.
- Remember that warps all happen at the same time
  - Loads for all threads happen at the same time
  - If there is any divergence, then it is duplicated across all threads, but masked off
- `Registered equipment` takes no time at all for soldiers to use. Private to each thread. We want as much of this as we can without killing occupancy
  - Drill seargants understand that you're going to need to fetch and request equipment regularly, and of course tolerate an amount of it. But if you get lazy and keep requesting for things from global memory you could have computed once and stored on your person as a `registered weapon`, then the drill seargant/central command will be pissed at you - even though warp sheduling drill seargants can't be angry.
- `Shared memory` is equipment brought by each admin team by their `block ship`.
  - Its supposed to be handled explicitly by the `coder central command`. But it really helps if I can specify ahead of time to the `compiling officials` how big its expected to be.
  - Shared memory is on chip and I manage it.
  - If I do specify how big this is at `compile time of the plans`, then more efficient and less commands can be issued about interfacing with it, more unrolling and less address arithmetic.
  - If a soldier needs something from here, it is 20 cycles or so waiting, which is 5 times slower than ALU arithmetic it could be doing. Memory is EXPENSIVE.
- `L1/TEX` is the storage cupboard shared by the `entire SM port`, and its only twice as slow as shared memory. But still significant
  - This is hardware controlled now, so not something I control directly anymore outside of proper code design.
  - Serves many small request
- `L2`
  - A frontal cache shared between the entire military port
  - These arrive in some kind of pneumatic tube that still takes a long time, the tube movement is fast, but decoding where exactly this one soldier who request is takes a while and the requested item moves through a few places before arriving
- `DRAM`
  - Also shared between the entire military port
  - 4 times slower at least than even the L2 cache
  - You'll really be stood here a while. Really try not to request this, and if your warp squad must do this, at least make it easy on the depot by asking for them in an unrolled loop with no dependencies, and coalesced preferably.

### Divergence, Coalescing, and Burst Loading
Generally, memory systems charge a large fixed `setup cost` per request, but a much smaller incremenetal cost be additional item at L1, L2, and DRAM.

#### Divergence
```cpp
if (tidx < 16)
    do_A();
else
    do_B();
```
- Phase 1: Warp executes `do_A()` together
  - `Soldiers 0-15` actively do it
  - `Soldiers 16-31` don't hear the order, and their outputs are masked as they are irrelevant
- Phase 2: Afterwards warp executes `do_B()` together
  - `Soldiers 16-31` actively do it
  - `Soldiers 0-15` don't hear the order, and their outputs are masked as they are irrelevant
 
### Coalescence and Burst loading
- `Coalescing`
  - Is about asking for contiguous memory regions
  - When the memory load is executed by the `thread soldiers` in the `warp squadron` altogether at the same time, the addresses are all considered at once and broken down into the fewest contiguous reads possible.
  - If i've designed the `kernel instructions` well, we'll get a nice `0-31` for a given memory region, reducing overheads in the read and allowing address reuse or some such.
- `Burst loading`
  - Given a bunch of things all ready to fill the delivery box or tube. Burst loading is the tendency for the depot staff to be very fast at filling the pneumatic tube it will send back to the warp squad with individual items.
  - This tends to happen at the same time as coalesced requests.


## Weakness with the old analogy to resolve:
- Very saturated blocks are NOT like a ship near full capacity of soldiers
  - Theres nothing stopping a full ships just docking and waiting for their many soldiers
  - Blocks in an SM don't work like that, if they are full of warps they don't just "wait their turn". There is a limit somehow on block size per SM, regardless of if the block takes very little registers or shared memory. Why is this?
    - Because a resident block isn't just a collection of threads. Its a *commitment* that is given saying "every thread and register and address here exists and will not require a context switch"
    - We don't want expensive context switches on GPUs, they're designed for raw throughput and not flexibility
    - Consider `__syncthreads()`, if we allowed threads that didn't quite exist yet, a single call to this may stall if the thrad never existed, leaving a dead block.
  - As for partially unloaded ships or only letting out a few at once breaking the analogy, I'll let `__syncthreads()` save me here:
      - Valid military preparations require everyone on the block ship to be reachable at all times. Literally every single soldier's preparation must be accounted for. The software that gives them orders they synchronise to the moment they leave the ship requires every soldier to sign up to it at the same time so work scheduling can begin. If any single soldier is unaccounted for or not yet synchronised with his team, his crucial contribution to their shared work cannot continue and will stall forever. The docks COULD have been designed to allow for adding soldiers later, essentially forcing the need for a CPU-lke context switch under certain circumstances. However, the dock has been designed with efficiency in mind, and has chosen to be ruthless about its synchronisation such that the assumptions it allows them to can increase efficiency.
      - An improved version according to chatGPT:
        - A block is a sealed military unit whose command software assumes that every soldier exists, is registered, and can be synchronised with at any moment.
        - Orders may include “wait until everyone is ready” commands.
        - If even one soldier is not present, the unit can stall forever.
        - Therefore, the dock refuses to admit a unit unless it can unload everyone, all equipment, and all communications infrastructure immediately.
        - My addition: Again, the dock could have been designed to allow late starters, but this would require allowing CPU-like context switches and hardware/clock speed/efficiency concessions that we elect to forgo for the GPU mantra of trading flexibility for efficiency.


## Load Imbalance
- When you have some kind of for loop say iterating in a warp of work, and the total iterations across each thread isn't a clean multiple of 32, then some threads will end up doing work that others don't need to
- If this can't be helped, then it is what it is, but this is load imbalance and it can be introduced accidentally quite easily
- Often you can adjust hyperparameters like `TK` in tiled matmul to make it work
- The `kernel instructions` from a `warp squadron` may end up being a number that isn't 0 modulo 32. This will mean that `thrad soldiers` will be doing an iteration or so more work than their squadmates.


## Shared Memory - Bank Conflicts
So mechanically how shared memory works is pretty awesome. Don't so much think of it as a giant contiguous block of memory, but realise that it was designed to be accessed fast by warps.
- Shared memory has 32 banks that can read a floats worth of bits per bank per cycle. i.e. 32 bits for each 32 bank.
  - This is striped across memory not in chunks
  - `float 0 -> bank0`
  - `float 1 -> bank1`
  - ...
  - `float 31 -> bank31`
  - `float 32 -> bank0`
  - `float 33 -> bank1`
  - If you respect banking properly, then all 32 threads can receive something in 1 cycle:
    - `shmem[threadIdx.x];`
    - however, if all threads access crucially the SAME elemnt:
      - `shmem[0];`
      - It still comes out of the 0 bank, and you need to read the same bank for 32 cycles, one at a time
      - These are bank conflicts
- Bank conflict analogy:
  - The `shared memory logistics block` has `32 bank messengers in striped uniforms` that turn up at once to respond to reads
  - Each messenger must sign each order given from its pile with its own name
  - If each `thread soldier` in the `warp squadron` needs something from `bank messenger 0`, you're going to have to wait 32 cycles for them all to get it.
- This does indeed happen in 1 cycle
- But remember the shared-memory overhead of 20-30 cycles
  - This doesn't contradict the bank speed because we need to do things like address decoding for any given read. So we end up with 20-30 cycles before the read is issued and the data becomes visible to the warps still


## Compile-time vs runtime knowledge
- I write the `kernel instructions`, and the compiler turns them into actual logistics policies.
- If I am specific with numbers and not overly ambiguous, this allows the `logistics compilers` to enact certain efficienies:
  - Shared memory:
    - Know `shared memory` size ahead of time allows the `compiler logisitcs` to set out a `static arrayed area` with labels and other such that make things faster. Otherwise it must be an ambiguously large space, requiring sorting on the way in
    - `<Templating>` my `kernel` instructions makes my orders `readable, re-usable` by others, though the version i hand the `logisitcs compilers` squad should be specific about the numbers needed. It costs very little extra to have `compiled in multiple compile-time` copies of `kernel's orders` into the plans given that `executable` bag space isn't a concern.
    - `#pragma unroll` removes dependency chains and following loop overheads. i.e. not needing to wait for needless `shared memory` logic and just getting on with orders without needing to `unpack each with needless wrapping overhead`


## The purpose of CUDA kernel code
- Lots of squads walking around `occupying` my floor space is NOT the goal, but it is essential for hiding latency.
- Poorly planned, my `SM base` can keep itself busy `re-deriving the same addresses needlessly`. This is a failure.
- My job as `kernel` (lol) of this army base is to at a minimum complete the mission
- But actually my job is to learn from the first few runs I try, and adapt early to make y instructions the most efficient way possible.
- I then ship these out for others to use.
- I am an efficiency optimiser, I am not an end user.


## Naive Kernels
- Occam's razor has some place on this here base
- Caching and such can mask bad access patterns and floor space, getting you 80% of the way there
- Naive overengineering can practically slow things down, as the `logisitcs compilers` try to work my orders
- But my job isn't mission success.
  - My minimal job is a plan for AT LEAST mission success
  - Then to make the most efficient possible `kernel instructions`
  - This plan will be deployed to millions of bases, and resources are tight out there.




## Examples
```cpp
// Bad
__global__ void bad(const float* A, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += A[idx] * i;
    }
    out[idx] = acc;
}


// Good
__global__ void good(const float* A, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float a = A[idx];   // one load
    float acc = 0.0f;

    for (int i = 0; i < 32; ++i) {
        acc += a * i;
    }
    out[idx] = acc;
}
```

## CUDA Software:
- `Nvidia Driver`
  - The operating system of the GPU
  - Must be installed on any computer wanting to use nvidia GPU on any level
  - Also provides for Vulkan and Direct3D
  - Driver version numbers e.g. `r580`
- `CUDA Toolkit`
  - Set of libraries, headers, tools for writing buildig and analysing software
  - I think of it like an SDK from a board? I might be wrong
  - `CUDA Runtime`:
    - A special case of one libraries in the toolkit
    - Provides API and language extensions for common things like memcpy and malloc between devices and launching kernels.
    - I Import this header all the time!

## Resources from CUDA
- [CC table per hardware](https://developer.nvidia.com/cuda/gpus)
  - [Memory info per CC](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-table-memory-information-per-compute-capability)
- 