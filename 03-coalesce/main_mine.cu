#include <cuda_runtime.h>
#include <isostream>

#define CK(call) do {
    cudaError_t _e = (call);
    if (_e != cudaSuccess {
        std::cerr << "CUDA error: " << cudaGetErrorString(_e) << "at" << __FILE__ << ";" << __LINE__ << std::endl;
        std::exit(1);
    }
} while(0)


int main() {

}