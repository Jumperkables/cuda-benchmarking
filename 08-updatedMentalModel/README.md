# New Mental Model
- Check the accompanying memory_throughput md file
- Its time to update my military mental model outlined in [section 03](../03_mental-model/README.md)

## A new analogy for memory movement - Building on the military one
Ok, i stumbled on the need for a memory ordering analogy that I'll need to get right first before the other ones
Why all code doesn't do the same thing:
- Cache hits and misses differ
- Address dependencies and decoding overhead

### Rules of thumb
- Register: ~1 cycle
  - Private to each thread
  - Each soldier carries and is given these, no sharing.
- Shared mem: ~20-30 cycles
  - Bunker cache brought in by the admin team. The whole block shares this.
- L1/TEX (Texture): 30-50 cycles
  - Warehouse for the 1 port
  - Shared per SM

**Shared by ALL SMs**
- L2: ~100 cycles
  - Base storage center
  - Hardware managed
- DRAM: ~400-800 cycles
  - Back in home country
  - Off chip
  - High latency, high bandwidth, so we want many requests at once here

### Analogy:
- Each cycle in different memory movement is made of essential steps that are all quite fast in themselves. i.e. a 400 cycle DRAM read is NOT just 399 cycles of prep and 1 cycle of movement, its more like very few steps each quite expensive due to off-chip travel, queuing, arbitration,and DRAM row activations.
- Remember that warps all happen at the same time
  - Loads for all threads happen at the same time
  - If there is any divergence, then it is duplicated across all threads, but masked off
- `Registered equipment` takes no time at all for soldiers to use. Private to each thread. We want as much of this as we can without killing occupancy
  - Drill seargants understand that you're going to need to fetch and request equipment regularly, and of course tolerate an amount of it. But if you get lazy and keep requesting for things from global memory you could have computed once and stored on your person as a `registered weapon`, then the drill seargant/central command will be pissed at you - even though warp sheduling drill seargants can't be angry.
- `Shared memory` is equipment brought by each admin team by their `block ship`.
  - Its supposed to be handled explicitly by the `coder central command`. But it really helps if I can specify ahead of time to the `compiling officials` how big its expected to be.
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


## Things to add to the analogy:
- `__syncthreads()`
- Latency hiding is kind of a thing only because ALUs are shared. If every warp had its own ALU, then WARP B wouldn't be able to hide its latency while it waits for memory. Also instruction fetch, and memory pipelines.
## ChatGPTs topic recommendations:
Core mental model to harden - Think in three layers simultaneously:
- Hardware reality (what physically exists)
- Compiler knowledge (what can be planned in advance)
- Runtime behavior (what actually happens during execution)

Your analogy should map each concept to one of these layers.
1) Latency hiding:
- What you learned
  - Global memory is slow, but GPUs don’t wait for it.
  - Warps are swapped in/out to hide latency.
  - Occupancy is not a goal; it’s a means to hide latency.
  - High occupancy + dependency chains = still slow.
- Military analogy
    - A single squad waiting for supplies is useless.
    - A base runs many squads; while one waits, others fight.
    - Latency hiding = always having another squad ready.
- Key insight to harden
  - The GPU does not make memory faster — it avoids waiting.
- Tie this to:
  - warps per SM
  - dependency chains (acc += ...)
  - why unrolling helps (multiple independent instructions)

2) Shared memory
- What you learned
  - Shared memory is on-chip, fast, and cooperative.
  - It is not a cache; it’s a manually managed scratchpad.
  - Static shared memory enables compiler optimization.
  - Dynamic shared memory trades performance for flexibility.
- Military analogy
  - Shared memory = ammo cache inside the bunker.
  - Global memory = supply depot miles away.
  - Static shared = prebuilt bunkers with fixed layout.
  - Dynamic shared = temporary tents you assemble on arrival.
- Key insight to harden
  - Shared memory is powerful because it turns many long trips into one short trip — but only if organized well.
- Tie this to:
  - tiling
  - reuse across K
  - cooperative loads
  - bank conflicts (friendly fire inside the bunker)

3) `__syncthreads()`
- What you learned
  - It is both an execution barrier and a memory fence.
  - It is not inherently slow; waiting is slow.
  - It exposes imbalance and divergence.
  - You need it whenever threads depend on each other’s data.
- Military analogy
  - __syncthreads() = “hold position until all squads arrive.”
  - If one squad is late, everyone waits.
  - The order itself isn’t costly; the straggler is.
- Key insight to harden
  - Barriers don’t cause slowness — uneven work does.
- Tie this to:
  - load imbalance in loops
  - warp-level vs block-level coordination
  - why tiling must be symmetric

4) Compile-time vs runtime knowledge
- What you learned
  - The compiler is your logistics planner.
  - If it knows sizes at compile time, it:
    - unrolls loops
    - erases address math
    - schedules instructions better
  - Runtime variability prevents those optimizations.
- Military analogy
  - Compile-time constants = pre-war planning.
  - Runtime values = battlefield improvisation.
  - Planned operations beat improvised ones.
- Key insight to harden
  - Performance comes from what the compiler can prove, not what you know as a human.
- Tie this to:
  - templates
  - static shared arrays
  - #pragma unroll
  - why dynamic kernels underperform
  - Why we want a dynamic fallback in the first place

5) Divergence
- What you learned
  - A warp executes as one unit.
  - If threads take different paths, they serialize.
  - Divergence wastes lanes, not time per se.
- Military analogy
  - A platoon moves together.
  - If half go left and half go right, they must take turns.
  - Empty trucks still burn fuel.
- Key insight to harden
  - Divergence turns parallel units into serial ones.
- Tie this to:
  - conditionals on tx, ty
  - edge guards
  - why padding and masking exist

6) Load imbalance
- What you learned
  - Even without divergence, unequal work causes waiting.
  - Load imbalance shows up at barriers.
  - Dynamic loops often create imbalance.
- Military analogy
  - Some squads finish fast, others slog through mud.
  - Everyone waits at rendezvous.
  - The slowest squad defines the pace.
- Key insight to harden
  - Balanced work is more important than fast individual threads.
- Tie this to:
  - cooperative loads
  - equal iteration counts
  - tile shapes

7) Instruction overhead vs useful work
- What you learned
  - High SM utilization ≠ high FLOP/s.
  - The SM can be busy doing “bookkeeping.”
  - Dynamic indexing, loops, and sync inflate instruction count.
- Military analogy
  - Soldiers constantly moving doesn’t mean progress.
  - Marching, relaying orders, waiting all burn energy.
  - Only firing at the enemy counts.
- Key insight to harden
  - You optimize by increasing useful work per instruction.
- Tie this to:
  - FLOPs per instruction
  - unrolling
  - register reuse

8) Why naive kernels can look “surprisingly good”
- What you learned
  - Caches can mask bad access patterns.
  - Moderate sizes sometimes fit well in L2.
  - Naive kernels are simple and low-overhead.
- Military analogy
  - Small battles don’t need complex logistics.
  - Overengineering can slow things down.
- Key insight to harden
  - Optimization only wins once overhead is amortized.

9) Professional CUDA workflow (this is important)
- What you implicitly learned
  - Write a correct dynamic baseline
  - Write one specialized fast path
- Dispatch based on known shapes
  - Compare bytes, FLOPs, and time
  - Iterate
- Military analogy
  - Train recruits (dynamic)
  - Field elite units (specialized)
  - Deploy appropriately