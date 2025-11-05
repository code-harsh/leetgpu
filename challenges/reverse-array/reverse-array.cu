#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (N + 1) / 2)
    {
        float tmp;
        tmp = input[idx];
        input[idx] = input[N - idx - 1];
        input[N - idx - 1] = tmp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

/*
0 n-1
1 n-2 
...
...
i n-i-1

n = 5 -> iterate 0 1 2 and swap
n/2 = 2 
*/
