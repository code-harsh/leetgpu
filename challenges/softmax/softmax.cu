#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, float sum, float max, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {    
        output[idx] = expf(input[idx] - max) / sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float max = -INFINITY;
    for (int j = 0; j < N; j++) {
        max = fmaxf(max, input[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += expf(input[j] - max);
    }
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, sum, max, N);
    cudaDeviceSynchronize();
}