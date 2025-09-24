# Benchmarking screenPixelExample

- I made a dummy operation of a tensor that has roughly the resolution of 4 computer screens (1920, 1080, 4)
    - Doubles each 'pixel' value for no particular reason
    - Notably, the tensor was a bit small for proper benchmarking (most of it was just the memcpy overhead i think)
    - So I increased the tensor size to (19200, 10800, 4)
- Learned how to make multidimensional block and grid logic, awesome. Seems like the building blocks to a lot of CUDA
- Block sizes are at max 1024 (harmonises with hardware constraints for the streaming multiprocessors)

## Block configurations I tried:
| `x`  | `y`  | `z` | `~time (ms)` |
|------|------|-----|--------------|
| 1024 | 1    | 1   | 710          |
| 512  | 2    | 1   | 343          |
| 512  | 1    | 1   | 233          |
| 256  | 2    | 1   | 116          |
| 256  | 1    | 1   | 136.5        |
| 128  | 4    | 1   | 60.3         |
| 64   | 8    | 1   | 34.1         |
| 32   | 16   | 1   | 20.2         |
| 32   | 1    | 1   | 140          |
| 16   | 32   | 1   | 15.0         | 
| 16   | 32   | 2   | 19.2         | 
| 16   | 64   | 1   | 19.5         |
| 8    | 64   | 1   | 17.2         |
| 4    | 128  | 1   | 17.8         |
| 2    | 256  | 1   | 20.0         |
| 1    | 512  | 1   | 44.8         |

## Explaining these benchmark results
Most of these make sense with what you'd expect intuitively, a combination of online guides and ChatGPT assure me that:

### Oversizing x is bad (1024, 1, 1)
At first having a large x for such a big task made sense as everything would be contiguous or close to eachother in memory. But remember that the maximum block size is 32, and there are 32 warps.
- So first I went and learned typical GPU and CUDA architecture for about 3 hours
- Typically there are 4 warp schedulers per block
- With a maximum thread count of 2048 on the SM
- If there is an oversized block e.g. 1024, then only 2 of them fit in the SM at once
- This means that only 2 of the schedulers are working on average
- The sweet spot range of 256 - 512 threads is
  - 512 = 4 blocks which is the same number as typical warp schedulers
  - 256 = 8 blocks, which can be interleaved and chosen more agile than bigger ones
  - Having too small a block size incurrs th overhead of additional blocks too often, slowing things down again
- So the improvment from (1024,1,1) -> (512,1,1) is likely a lot to do with warp scheduler occupancy

# I probably messed up coalescence
- However, the poor results in this means that I have accidentally got my coalescing the wrong way around
- It should not be the case that (512,1,1) is slower than (1,512,1)
- If i look into this, I will likely find a bug that implies i'me coalescing column major by accident instead of row major