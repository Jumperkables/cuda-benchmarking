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
