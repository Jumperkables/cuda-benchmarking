# New Mental Model
- Check the accompanying memory_throughput md file

## ChatGPTs topic recommendations:
Core mental model to harden

Think in three layers simultaneously:

Hardware reality (what physically exists)

Compiler knowledge (what can be planned in advance)

Runtime behavior (what actually happens during execution)

Your analogy should map each concept to one of these layers.

1) Latency hiding

What you learned

Global memory is slow, but GPUs don’t wait for it.

Warps are swapped in/out to hide latency.

Occupancy is not a goal; it’s a means to hide latency.

High occupancy + dependency chains = still slow.

Military analogy

A single squad waiting for supplies is useless.

A base runs many squads; while one waits, others fight.

Latency hiding = always having another squad ready.

Key insight to harden

The GPU does not make memory faster — it avoids waiting.

Tie this to:

warps per SM

dependency chains (acc += ...)

why unrolling helps (multiple independent instructions)

2) Shared memory

What you learned

Shared memory is on-chip, fast, and cooperative.

It is not a cache; it’s a manually managed scratchpad.

Static shared memory enables compiler optimization.

Dynamic shared memory trades performance for flexibility.

Military analogy

Shared memory = ammo cache inside the bunker.

Global memory = supply depot miles away.

Static shared = prebuilt bunkers with fixed layout.

Dynamic shared = temporary tents you assemble on arrival.

Key insight to harden

Shared memory is powerful because it turns many long trips into one short trip — but only if organized well.

Tie this to:

tiling

reuse across K

cooperative loads

bank conflicts (friendly fire inside the bunker)

3) __syncthreads()

What you learned

It is both an execution barrier and a memory fence.

It is not inherently slow; waiting is slow.

It exposes imbalance and divergence.

You need it whenever threads depend on each other’s data.

Military analogy

__syncthreads() = “hold position until all squads arrive.”

If one squad is late, everyone waits.

The order itself isn’t costly; the straggler is.

Key insight to harden

Barriers don’t cause slowness — uneven work does.

Tie this to:

load imbalance in loops

warp-level vs block-level coordination

why tiling must be symmetric

4) Compile-time vs runtime knowledge

What you learned

The compiler is your logistics planner.

If it knows sizes at compile time, it:

unrolls loops

erases address math

schedules instructions better

Runtime variability prevents those optimizations.

Military analogy

Compile-time constants = pre-war planning.

Runtime values = battlefield improvisation.

Planned operations beat improvised ones.

Key insight to harden

Performance comes from what the compiler can prove, not what you know as a human.

Tie this to:

templates

static shared arrays

#pragma unroll

why dynamic kernels underperform

5) Divergence

What you learned

A warp executes as one unit.

If threads take different paths, they serialize.

Divergence wastes lanes, not time per se.

Military analogy

A platoon moves together.

If half go left and half go right, they must take turns.

Empty trucks still burn fuel.

Key insight to harden

Divergence turns parallel units into serial ones.

Tie this to:

conditionals on tx, ty

edge guards

why padding and masking exist

6) Load imbalance

What you learned

Even without divergence, unequal work causes waiting.

Load imbalance shows up at barriers.

Dynamic loops often create imbalance.

Military analogy

Some squads finish fast, others slog through mud.

Everyone waits at rendezvous.

The slowest squad defines the pace.

Key insight to harden

Balanced work is more important than fast individual threads.

Tie this to:

cooperative loads

equal iteration counts

tile shapes

7) Instruction overhead vs useful work

What you learned

High SM utilization ≠ high FLOP/s.

The SM can be busy doing “bookkeeping.”

Dynamic indexing, loops, and sync inflate instruction count.

Military analogy

Soldiers constantly moving doesn’t mean progress.

Marching, relaying orders, waiting all burn energy.

Only firing at the enemy counts.

Key insight to harden

You optimize by increasing useful work per instruction.

Tie this to:

FLOPs per instruction

unrolling

register reuse

8) Why naive kernels can look “surprisingly good”

What you learned

Caches can mask bad access patterns.

Moderate sizes sometimes fit well in L2.

Naive kernels are simple and low-overhead.

Military analogy

Small battles don’t need complex logistics.

Overengineering can slow things down.

Key insight to harden

Optimization only wins once overhead is amortized.

9) Professional CUDA workflow (this is important)

What you implicitly learned

Write a correct dynamic baseline

Write one specialized fast path

Dispatch based on known shapes

Compare bytes, FLOPs, and time

Iterate

Military analogy

Train recruits (dynamic)

Field elite units (specialized)

Deploy appropriately