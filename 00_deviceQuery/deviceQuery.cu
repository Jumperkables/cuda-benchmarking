#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("=======================================================\n");
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("=======================================================\n");

        // Identification
        printf("Compute Capability:                 %d.%d\n", prop.major, prop.minor);
        printf("UUID:                               ");
        for(int i = 0; i < 16; ++i) printf("%02x", (unsigned char) prop.uuid.bytes[i]);
        printf("\n");

        // Core architecture / resources
        printf("Streaming Multiprocessors (SMs):    %d\n", prop.multiProcessorCount);
        printf("Warp Size:                          %d\n", prop.warpSize);
        printf("Max Threads / SM:                   %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max Threads / Block:                %d\n", prop.maxThreadsPerBlock);
        printf("Max Block Dimensions:               %d x %d x %d\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions:                %d x %d x %d\n",
                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // Registers
        printf("Registers per Block:                %d\n", prop.regsPerBlock);
        printf("Registers per SM:                   %d\n", prop.regsPerMultiprocessor);

        // Shared memory
        printf("Shared Memory per Block:            %zu bytes\n", prop.sharedMemPerBlock);
        printf("Shared Memory per SM:               %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("Total Constant Memory:              %zu bytes\n", prop.totalConstMem);

        // Memory subsystem
        printf("Global Memory:                      %zu bytes\n", prop.totalGlobalMem);
        printf("Memory Bus Width:                   %d bits\n", prop.memoryBusWidth);

        // Cache info
        printf("L2 Cache Size:                      %d bytes\n", prop.l2CacheSize);

        // ECC, async engines, concurrency
        printf("Concurrent Kernels:                 %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("ECC Enabled:                        %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("Async Engine Count (DMA engines):   %d\n", prop.asyncEngineCount);
        printf("Unified Addressing:                 %s\n", prop.unifiedAddressing ? "Yes" : "No");


        // Texture and surface limits
        printf("Max Texture 1D Size:                %d\n", prop.maxTexture1D);
        printf("Max Texture 2D Size:                %d x %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
        printf("Max Texture 3D Size:                %d x %d x %d\n",
               prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);

        printf("\n");
    }

    return 0;
}
