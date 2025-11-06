# LeetGPU Solutions

[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A collection of optimized CUDA solutions for [LeetGPU](https://leetgpu.com/challenges) programming challenges. This repository showcases GPU-accelerated implementations of various algorithmic problems, demonstrating efficient parallel computing techniques.

## Progress Overview

📊 Current Progress ([Profile](https://leetgpu.com/profile?display_name=mad_scientist)):
- Easy: 15/15 completed
- Medium: 0/34 completed
- Hard: 0/8 completed

## Environment

- NVIDIA Tesla T4 GPU
- CUDA Toolkit 12.x
- Tested on LeetGPU platform

## CUDA Quick Reference

### Grid and Block Indexing

#### 1D Arrays
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

#### 2D Matrices
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * cols + col;
```

#### 3D Volumes
```cuda
int z = blockIdx.z * blockDim.z + threadIdx.z;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
int idx = z + D3 * (y + x * D2);
```

## Contributing

Feel free to contribute by:
1. Solving new problems
2. Optimizing existing solutions
3. Improving documentation
4. Reporting issues

## Work in Progress

This repository is actively maintained and updated. Solutions are continuously optimized as we explore more advanced CUDA techniques and patterns.
