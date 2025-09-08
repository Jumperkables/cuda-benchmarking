
# 02 — SAXPY

## What I Did
- Learned what blocks and grids are, and how they relate to hardware (warps and SMs) at a super basic level
- 

## 📊 Results
| N (elements) | Block Size | CPU Time (ms) | GPU Time (ms)         | Speedup (×) |
|--------------|------------|---------------|-----------------------|-------------|
| 1<<30        | 32         | ~270          | 20.4                  |             |
| 1<<30        | 128        | ~270          | 15.4                  |             |
| 1<<30        | 512        | ~270          | 15.7                  |             |
| 1<<30        | 1024       | ~270          | 15.5                  |             |
| 1<<30        | 4096       | ~270          | Invalid conf argument |             |


## Places I messed up
- 

## Lessons Learned
- "What is block size?"
  - Instead of starting some amount of threads, you organise threads into a grid of blocks
  - **Thread:** Runs the kernel code on 1 element. Usually a given warp (hardware collation of threads) is 32 threads.
  - **Block:** A group of threads, that can apparently be 1D, 2D, or 3D. Threads in the same block can share fast memory and sync
  - **Grid:** A collection of blocks which together cover the entire problem
- CUDA gives blocks shared memory between all threads inside it.
- Pick a multiple of 32 for your block size to not waste threads in a warp
- Try to be conscious of shared memory size you'll be getting per block. Streaming multiprocessors (collectiosn of warps) have a fixed memory and we want to fit blocks inside efficiently.
- For now i shouldn't really be thinking about this. I should just be learning about coalescence.

## Environment
- **GPU(s):** RTX 3090 / RTX 5060 Ti (Blackwell)  
- **CUDA Toolkit:** 12.9  
- **OS:** Arch Linux 
- **Compiler:** nvcc -O3

## Next Steps
- Extend to **SAXPY** (y = a*x + y) and measure memory coalescing effects.
- Explore larger N values and profile with Nsight for more detail.
- Compare behavior on both GPUs (Ampere vs Blackwell).
