#include <cuda_runtime.h>

__global__ void convolution2d(const float* input, const float* kernel, float* output,
                          int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    if (out_row < output_rows && out_col < output_cols) {
        float sum = 0.0f;
        for (int kr = 0; kr < kernel_rows; ++kr) {
            for (int kc = 0; kc < kernel_cols; ++kc) {
                sum += input[(out_row + kr) * input_cols + (out_col + kc)] *
                       kernel[kr * kernel_cols + kc];
            }
        }
        output[out_row * output_cols + out_col] = sum;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // Define block and grid sizes       
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    dim3 blockSize(16, 16);
    dim3 gridSize((output_cols + blockSize.x - 1) / blockSize.x,
                  (output_rows + blockSize.y - 1) / blockSize.y);
    // Launch kernel
    convolution2d<<<gridSize, blockSize>>>(input, kernel, output,
                                      input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}