# Mental Model for Understanding Memory Throughput - L1, L2, DRAM, and thread indexing 
By analysing my naive GEMM benchmark properly, the aim here is to crystallise a strong mental model of:
- What exactly is going on in memory
- Why the kernel design of naive matmul leads to this
- What kinds of kernel code design decisions will mitigate this in the future