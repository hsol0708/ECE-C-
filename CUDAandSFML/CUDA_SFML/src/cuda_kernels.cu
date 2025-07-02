// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__device__ int countNeighbors(const char* currentGrid, int x, int y, int GRID_WIDTH, int GRID_HEIGHT) {
    int count = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0) continue;
            int nx = (x + i + GRID_WIDTH) % GRID_WIDTH;
            int ny = (y + j + GRID_HEIGHT) % GRID_HEIGHT;
            count += currentGrid[ny * GRID_WIDTH + nx];
        }
    }
    return count;
}

__global__ void updateGridKernel(char* grid, char* newGrid, int GRID_WIDTH, int GRID_HEIGHT) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < GRID_WIDTH && y < GRID_HEIGHT) {
        int idx = y * GRID_WIDTH + x;
        int neighbors = countNeighbors(grid, x, y, GRID_WIDTH, GRID_HEIGHT);

        if (grid[idx]) {
            newGrid[idx] = (neighbors == 2 || neighbors == 3);  // Cell stays alive
        } else {
            newGrid[idx] = (neighbors == 3);  // Cell becomes alive
        }
    }
}

extern "C" void updateGrid(char* d_currentGrid, char* d_newGrid, int GRID_WIDTH, int GRID_HEIGHT, dim3 grid, dim3 block) {
    updateGridKernel<<<grid, block>>>(d_currentGrid, d_newGrid, GRID_WIDTH, GRID_HEIGHT);
    cudaDeviceSynchronize();  // Ensure kernel execution completes
}

void vectorAdd(const float* A, const float* B, float* C, int N) 
{
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "blocksPerGrid = " << blocksPerGrid << std::endl;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}