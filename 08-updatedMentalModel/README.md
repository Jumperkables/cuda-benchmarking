# New Mental Model
- Check the accompanying memory_throughput md file
- Its time to update my military mental model outlined in [section 03](../03_mental-model/README.md)

# A new analogy for memory movement - Building on the military one
Ok, i stumbled on the need for a memory ordering analogy that I'll need to get right first before the other ones
Why all code doesn't do the same thing:
- Cache hits and misses differ
- Address dependencies and decoding overhead

## Rules of thumb
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

## Analogy:
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