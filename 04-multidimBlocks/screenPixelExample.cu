#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>
using namespace std;


#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while(0)


__global__ void screenPixelExample(float* pixTensor, int numThreads, int X, int Y, int Z){
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    int y = threadIdx.y + (blockIdx.y * blockDim.y);
    int z = threadIdx.z + (blockIdx.z * blockDim.z);
    if (x < X && y < Y && z < Z){
        int idx = x + (y*X) + (z*X*Y);
        pixTensor[idx] *= 2;
    }
}


int main() {
    // A practice program to get used to 3d indexing
    int dim_x = 19200;
    int dim_y = 10800;
    int dim_z = 4;
    int numThreads = dim_x * dim_y * dim_z;      // One for each pixel across my 4 screens say
    size_t bytes  = sizeof(float) * numThreads;  // Have a float for each pixel

    // Make some array that holds a big tensor for every pixel on the 'screen'
    vector<float> h_pixTensor(numThreads);
    for (int z=0; z<dim_z; z++){
        for (int y=0; y<dim_y; y++){
            for (int x=0; x<dim_x; x++){
                h_pixTensor[z*(dim_y*dim_x) + (y*dim_x) + x] = z;   // Each pixel should have its screen id on it
            }
        }
    }

    // Print sum total of tensor
    float preSum = std::accumulate(h_pixTensor.begin(), h_pixTensor.end(), 0.0f);
    std::cout << "Pre sum = " << preSum << "\n";

    // Make the device array, it has to be a null pointer to start with
    float *d_pixTensor;
    cudaMalloc(&d_pixTensor, bytes);
    cudaMemcpy(d_pixTensor, h_pixTensor.data(), bytes, cudaMemcpyHostToDevice);

    // Figure out the block size I want and how many threads to launch
    dim3 block(8, 32, 2);
    dim3 grid(
        (dim_x+block.x-1)/block.x,
        (dim_y+block.x-1)/block.y,
        (dim_z+block.x-1)/block.z
    );

    // Timing
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    CK(cudaEventRecord(start)); // Start timing the event?

    screenPixelExample<<<grid, block>>>(d_pixTensor, numThreads, dim_x, dim_y, dim_z);
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    }

    // output the elapsed time
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Kernel time: " << ms << "ms\n";

    // Move the array back to the host
    cudaMemcpy(h_pixTensor.data(), d_pixTensor, bytes, cudaMemcpyDeviceToHost);
    float postSum = std::accumulate(h_pixTensor.begin(), h_pixTensor.end(), 0.0f);
    std::cout << "Post sum = " << postSum << "\n";

    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));
    cudaFree(d_pixTensor);
}