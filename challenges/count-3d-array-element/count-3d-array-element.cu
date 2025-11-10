#include <cuda_runtime.h>

// Kernel to count elements greater than P
__global__ void count_p(const int* input, int* output, int totalElements, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        atomicAdd(output, (input[idx] == P ? 1 : 0));
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    int totalElements = N * M * K;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    count_p<<<blocksPerGrid, threadsPerBlock>>>(input, output, totalElements, P);
    cudaDeviceSynchronize();
}