
# 01 — Vector Add

## What I Did
- Implemented a baseline **CPU vector addition**.
- Wrote a **CUDA kernel** for vector addition with grid/block indexing and bounds checking.
- Allocated/copy host ↔ device memory.
- Timed CPU vs GPU using `std::chrono` and `cudaEvent` timers.
- Ran experiments with different vector sizes (`N = 1<<20`, `1<<22`, `1<<24`) and block sizes (`128, 256, 512`).

## 📊 Results
| N (elements) | Block Size | CPU Time (ms) | GPU Time (ms) | Speedup (×) |
|--------------|------------|---------------|---------------|------------|
| 1<<20        | 128        | 0.08512       | 0.05533       | ~1.6       |
| 1<<20        | 256        | 0.09022       | 0.05718       | ~1.6       |
| 1<<20        | 512        | 0.07457       | 0.04912       | ~1.6       |
| 1<<22        | 512        | 0.80679       | 0.09091       | ~10        |
| 1<<24        | 512        | 4.37062       | 0.42141       | ~10        |
| 1<<30        | 512        | 273.465       | 15.4046       | ~18        |

## Places I messed up
- Getting used to looking for the `;` delimiters. Been away from C++ for too long
- Catching the segfaults came from deleting the arrays by accident before giving it to the device code

## Lessons Learned
- How to compute thread index with `blockIdx`, `blockDim`, and `threadIdx`.
- Importance of **bounds checking** (`if (idx < N)`).
- How to time kernels with `cudaEvent_t` without including host↔device transfers.
- GPU speedup depends on both **problem size** (too small → GPU overhead dominates) and **block size** (128 vs 256 vs 512).
- First exposure to **error-checking macros** and catching CUDA launch errors.

## Environment
- **GPU(s):** RTX 3090 / RTX 5060 Ti (Blackwell)  
- **CUDA Toolkit:** 12.9  
- **OS:** Arch Linux 
- **Compiler:** nvcc -O3

## Next Steps
- Learn exactly what block and grid size are
- Extend to **SAXPY** (y = a*x + y) and measure memory coalescing effects.
- Explore larger N values and profile with Nsight for more detail.
- Compare behavior on both GPUs (Ampere vs Blackwell).
