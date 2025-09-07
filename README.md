# cuda-benchmarking

A collection of small but increasingly complicated CUDA kernels for my practice and improvement. Each contains the following:
- A CPU implementation for reference and comparison
- CUDA kernel variations
- Performance benchmarks, (time, bandwidth, GFlOps)
- Notes on what I learned, what worked, and what didn't

All experiments running on my machine will run on:
- Arch linux
- nvcc version 12.9.86
- Both my GPUs for comparison:
  - 3090
  - 5060-ti

## 📂 Projects

| # | Project | Description | Key Learnings |
|---|---------|-------------|---------------|
| 01 | Vector Add | Baseline CPU vs GPU vector addition | Kernel launch syntax, indexing, cudaEvent timing |
| 02 | SAXPY (Stride) | `y = a*x + y`, coalesced vs strided | Memory coalescing, block size effects |
| 03 | Matrix Transpose | Naive vs tiled/shared-memory | Shared memory, bank conflicts, tiling |
| 04 | Reduction | Sum with shared memory, warp shuffle | Parallel reduction, warp intrinsics |
| 05 | Histogram | Global atomics vs block-private | Contention, shared memory histogram |