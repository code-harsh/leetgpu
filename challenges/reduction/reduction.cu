#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(output, input[idx]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {  
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reduction_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}