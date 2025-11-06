#include <cuda_runtime.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {   
        // Sigmoid-Linear Unit (SiLU) activation silu(x) = x*sigmoid(x)
        float sigmoid_x = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = input[idx] * sigmoid_x;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

