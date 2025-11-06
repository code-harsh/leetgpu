#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < halfN)
    {
        // Sigmoid-Linear Unit (SiLU) activation silu(x) = x*sigmoid(x) for 1st half
        // SwiGLU(x1,x2) = SiLU(x1).x2
        float sigmoid_x = 1.0f / (1.0f + expf(-input[idx]));
        output[idx] = input[idx] * sigmoid_x * input[idx + halfN];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}