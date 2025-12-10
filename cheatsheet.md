# **CUDA Kernel Design Checklist**

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


## **Goals**

* Write kernels whose **instruction mix** and **memory behavior** allow ALUs and memory bandwidth to be well utilized.
* Ensure **enough active warps** to hide latency.
* Avoid unnecessary consumption of registers/shared memory that reduces resident warps or blocks.

---

# **1. Register Usage**

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

# **2. Block Size Selection**

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

# **3. Occupancy and Warp Count**

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

* Shared memory per block: **49,152 bytes** (48 KB)
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

## Military mental model I made
## [2025-12-10] Updated Mental Model - Military Operation
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
