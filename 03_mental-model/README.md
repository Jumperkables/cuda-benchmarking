# 03 Mental Model
I'm at the point where I can make my first mental model of GPUs and CUDA all in one spot. I'll make:
- A diagram
- A metaphor

That helps me understand whats going on here.

## Ballpark numbers
Its good to internalise a scale of how many SMs etc there are on various cards.
- See [my device query code](../00_deviceQuery) printing out my GPU info:

#### Most modern architectures
- Max number of warps / SM: 64
- Theoretical max number of warps / SM in practice: <64 (~48 or so)

#### 3090:
```
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
```

#### 5060ti:
```
Compute Capability:                 12.0
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
```

## Mental Model - Kitchen Analogy
- A GPU is a collection of kitchens making food
- I can already see this analogy won't be perfect, but it will be a good test of my knowledge and understanding to articulate where the analogy falls down, and what would be a more apt variant

**Hardware:**
- Each `SM` is a kitchen
- The 3090 food inc factory has `82` kitchens making food
  - The more kitchens, the more food you can make on average
- 4 `warp schedulers` in a kitchen. These are the head chefs.
- `Registers` are like tool slots near a table
  - Theres enough for `32` tool slots per WORKER. NOT per TABLE
    - Remember there are `32`

**Software:**
- The `kernel` i write is the type of food
  - `Simpler kernels` is like easy cook food - pasta
  - `Complex kernels` is like complicated food - burgers
- When i write a `kernel`, what I'm doing is writing instructions for food I want cooked thats going to be executed millions of times across all the kitchens
- `Blocks` are trucks of food that come in
- `Warps` are the boxes in the truck.
- `Threads` are the individual workers.

**Constraints:**
- `Always 32 threads in a warp`
  - Doesn't matter what the foodstuff, or how loaded the truck is. There are always 32 workers worth of food to prepare in each box. We underfill boxes if there aren't enough for a final box at the end.
- `Max threads / block = 2048`
  - `Blocks` only get so big because its impractical to make them bigger. The amount of work they would hold would become cumbersome
  - Trucks could be made huge. but realistically, we have a limited amount of space for parking. I don't know how to adapt this analogy further for why blocks aren't bigger
  - Each `block` can hold up to 2048 `threads` or `2048/32 = 64 warps`.
  - Each truck can hold up to 2048 foodstuffs, or at absolute most 64 boxes of food.
- `Max blocks / SM = 16`
  - Hard architectural limit
  - There are 16 parking spots outside each kitchen. Never more. Doesn't matter how lightly you load the trucks. 
- `Max hardware allowed warps active = 64`
  - There are 64 tables inside each kitchen
  - The maximum amount you can use at any one time depends on a few things

### Where the Kitchen analogy fails
- Actually, the threads are the workers, not the individual foodstuffs. I would repeat this analogy tomorrow to fix it.