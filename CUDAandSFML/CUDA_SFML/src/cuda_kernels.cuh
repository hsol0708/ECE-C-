// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

void vectorAdd(const float* A, const float* B, float* C, int N);
extern "C" void updateGrid(char* d_currentGrid, char* d_newGrid, int GRID_WIDTH, int GRID_HEIGHT, dim3 grid, dim3 block);
#endif
