# Tiled Matrix Multplication
Now that I understand the memory throughput problems with my naive matrix multiplication kernel, time to implement a tiled MM variation to address them.

# Interesting Learning:
- I learned you can template CUDA kernels
```cpp
template<int BM, int BN, int TK>
__global__ void tiled_MM(...){
    ...
}
```
- This templating here allows assumptions to be made at compilation time, not at run time
- Apparently this allows a bunch of benefits including:
  - Better loop unrolling
  - More optimal memory layout
  - Not needing to use extern for some reason

### Templating, Compile time variations, and Dynamic fallbacks
I've just learned that often in professional level CUDA code, people write dynamic fallbacks, and hard compile variations. So I'm going to start by making a dynamic fallback version. 
- template<typename T> DOESN'T pay by being unknown at compile time, the compiler figures it out
- But it DOES make the binary larger, which can be a problem