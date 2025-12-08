# 02 - Register and Shared Memory Pressure
- In `01_vector-add` I've already learned how to mess with block size and grid size to change occupancy
- Here, I'm going to intentionally mess with the code here to get used to how it changes performance
- Then next increasingly complicated lessons looking at stuff like coalescence I'll be able to monitor these things and figure out occupancy

## Lesson details
1. How to declare shared memory per block (static and dynamic)
2. How to modify your vector add to use shared memory
3. How to influence register usage (more vs less) in a simple, experimental way

You can then profile each variant with Nsight and see occupancy change.

---

## 1. Shared memory per block: how to declare it

There are two main patterns.

### A. Static shared memory

Size is known at compile time:

```cpp
__global__ void kernel(...) {
    __shared__ float buf[256];  // 256 * sizeof(float) bytes per block
    // each block gets its own buf[]
}
```

Key points:

* Every block has a **separate instance** of `buf`.
* The size counts against the per-SM shared memory budget.
* You do not pass the size at launch; it’s fixed in the code.

### B. Dynamic shared memory

Size is decided at launch time:

```cpp
__global__ void kernel_dynamic(const float* a, float* c, int n) {
    extern __shared__ float buf[];  // size determined at launch
    // buf has "sharedBytes / sizeof(float)" elements
}
```

Launch:

```cpp
size_t sharedBytes = block_size * sizeof(float);
kernel_dynamic<<<grid, block, sharedBytes>>>(d_a, d_c, n);
```

Key points:

* `extern __shared__` declares a **per-block** array whose size you specify in the third kernel launch parameter.
* All threads in the block see the same shared array.

---

## 2. Vector add with shared memory

Here’s a version of your vector add that uses dynamic shared memory. It’s not “better” than the direct version, but it’s perfect for learning shared memory and occupancy.

### Baseline kernel (no shared memory, low registers)

```cpp
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

This uses global memory directly, minimal registers, no shared memory.

### Shared-memory vector add (artificial, for learning)

```cpp
__global__ void vector_add_shared(const float* a, const float* b, float* c, int n) {
    extern __shared__ float sdata[];  // dynamic shared memory

    int tid  = threadIdx.x;
    int idx  = blockIdx.x * blockDim.x + tid;

    // Shared memory layout: [0..blockDim.x-1] -> a-tile
    //                       [blockDim.x..2*blockDim.x-1] -> b-tile
    float* s_a = sdata;
    float* s_b = sdata + blockDim.x;

    if (idx < n) {
        // load from global into shared
        s_a[tid] = a[idx];
        s_b[tid] = b[idx];
    }

    __syncthreads();  // ensure all loads complete

    if (idx < n) {
        float sum = s_a[tid] + s_b[tid];
        c[idx] = sum;
    }
}
```

Launch:

```cpp
dim3 block(block_size);
dim3 grid(ceil_div(n, block.x));

// need 2 * block_size floats of shared memory per block
size_t sharedBytes = 2 * block.x * sizeof(float);

vector_add_shared<<<grid, block, sharedBytes>>>(d_a, d_b, d_c, n);
```

Now each block uses:

* shared memory: `2 * blockDim.x * sizeof(float)`
* registers: slightly more than the baseline
* threads per block: unchanged

Profiling this kernel will show:

* per-block shared memory usage increased
* that may reduce max blocks per SM
* theoretical occupancy changes accordingly

You can vary `sharedBytes` (e.g. 4×, 8× blockDim.x) to see how it affects occupancy.

---

## 3. Controlling register usage (for experiments)

You do not directly “declare registers”, but the compiler allocates registers for:

* local variables
* temporary values
* unrolled loops
* local arrays that fit into registers

So you can influence register usage by changing the code.

### A. Forcing higher register usage (for experimentation)

Example:

```cpp
__global__ void vector_add_reg_heavy(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // lots of temporaries
    float x0 = a[idx];
    float x1 = x0 * 1.1f;
    float x2 = x1 * 1.1f;
    float x3 = x2 * 1.1f;
    float x4 = x3 * 1.1f;
    float x5 = x4 * 1.1f;
    float y0 = b[idx];
    float y1 = y0 * 0.9f;
    float y2 = y1 * 0.9f;
    float y3 = y2 * 0.9f;
    float y4 = y3 * 0.9f;
    float y5 = y4 * 0.9f;

    float result = (x5 + y5);
    c[idx] = result;
}
```

This is intentionally silly, but:

* it forces the compiler to keep many live values
* register count per thread goes up
* Nsight will show more registers/thread and potentially lower occupancy

If you compile with `nvcc --ptxas-options=-v`, you will see a line like:

```text
ptxas info    : Used 32 registers, 0 bytes smem, ...
```

You can compare this against the simpler kernel.

### B. Reducing register usage

You can lower register usage by:

* simplifying expressions
* avoiding large local arrays
* letting the compiler spill less
* or forcing a cap with a compile flag, e.g.:

  * `nvcc -Xptxas -maxrregcount=32 ...`

Capping registers can increase occupancy but may cause register spilling to local memory, which hurts performance. This is something you can observe in Nsight.

---

## 4. How to use this to build intuition

Concrete exercise:

1. Run and profile:

   * `vector_add` (baseline)
   * `vector_add_shared` with different shared memory sizes
   * `vector_add_reg_heavy` (higher register use)

2. For each, look at:

   * Registers per thread
   * Static + dynamic shared memory per block
   * Max blocks per SM
   * Theoretical and achieved occupancy
   * DRAM throughput and SM throughput

3. Change exactly one thing at a time (e.g. shared memory size) and see how one metric moves.

Doing this with a simple kernel like vector add is exactly the right way to make occupancy and resource constraints feel concrete instead of opaque.
