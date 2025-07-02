/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Sep 30,2024
Description: Lab 4: CUDA-based John Conway’s Game of Life
What is the purpose of this file?
Main file given from lecture
*/

#include <SFML/Graphics.hpp>
#include "cuda_kernels.cuh"
#include "commandline.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstring>

void seedRandomGrid(std::vector<char>& grid, int GRID_WIDTH, int GRID_HEIGHT) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i) {
        grid[i] = std::rand() % 2;
    }
}

void normalMemoryAllocation(char* h_currentGrid, char* h_newGrid, int GRID_WIDTH, int GRID_HEIGHT, int numThreads)
{
    char* d_currentGrid;
    char* d_newGrid;

    // Setup device memory
    cudaMalloc(&d_currentGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(char));
    cudaMalloc(&d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(char));
    cudaMemcpy(d_currentGrid, h_currentGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(char), cudaMemcpyHostToDevice);

    dim3 block(numThreads, numThreads);
    dim3 grid((GRID_WIDTH + block.x - 1) / block.x, (GRID_HEIGHT + block.y - 1) / block.y);

    for (int i = 0; i < 100; ++i) 
    {
        updateGrid(d_currentGrid, d_newGrid, GRID_WIDTH, GRID_HEIGHT, grid, block);
        cudaMemcpy(h_currentGrid, d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(char), cudaMemcpyDeviceToHost);
        std::swap(d_currentGrid, d_newGrid);
    }

    cudaFree(d_currentGrid);
    cudaFree(d_newGrid);
}

void pinnedMemoryAllocation(char* h_currentGrid, char* h_newGrid, int GRID_WIDTH, int GRID_HEIGHT, int numThreads)
{
    char* d_currentGrid;
    char* d_newGrid;
    char* pinned_currentGrid;
    char* pinned_newGrid;
    size_t size = GRID_WIDTH * GRID_HEIGHT * sizeof(char);

    cudaMallocHost(&pinned_currentGrid, size);
    cudaMallocHost(&pinned_newGrid, size);

    // Copy to pinned memory
    memcpy(pinned_currentGrid, h_currentGrid, size);

    cudaMalloc(&d_currentGrid, size);
    cudaMalloc(&d_newGrid, size);
    cudaMemcpy(d_currentGrid, pinned_currentGrid, size, cudaMemcpyHostToDevice);

    dim3 block(numThreads, numThreads);
    dim3 grid((GRID_WIDTH + block.x - 1) / block.x, (GRID_HEIGHT + block.y - 1) / block.y);

    for (int i = 0; i < 100; ++i) 
    {
        updateGrid(d_currentGrid, d_newGrid, GRID_WIDTH, GRID_HEIGHT, grid, block);
        cudaMemcpyAsync(pinned_newGrid, d_newGrid, size, cudaMemcpyDeviceToHost);
        std::swap(d_currentGrid, d_newGrid);
    }
    // Copy final grid to host
    memcpy(h_currentGrid, pinned_newGrid, size);

    cudaFree(d_currentGrid);
    cudaFree(d_newGrid);
    cudaFreeHost(pinned_currentGrid);
    cudaFreeHost(pinned_newGrid);

}

void managedMemoryAllocation(char* currentGrid, char* newGrid, int GRID_WIDTH, int GRID_HEIGHT, int numThreads) {
    size_t size = GRID_WIDTH * GRID_HEIGHT * sizeof(char);

    cudaMallocManaged(&currentGrid, size);
    cudaMallocManaged(&newGrid, size);

    cudaMemPrefetchAsync(currentGrid, size, 0);
    cudaMemPrefetchAsync(newGrid, size, 0);
    dim3 block(numThreads, numThreads);
    dim3 grid((GRID_WIDTH + block.x - 1) / block.x, (GRID_HEIGHT + block.y - 1) / block.y);

    for (int i = 0; i < 100; ++i) {
        updateGrid(currentGrid, newGrid, GRID_WIDTH, GRID_HEIGHT, grid, block);
        std::swap(currentGrid, newGrid);
    }

    cudaFree(currentGrid);
    cudaFree(newGrid);
}

int main(int argc, char* argv[]) 
{
   Arguments args = commandLineArguments(argc, argv);

    std::cout << "Parsed Arguments:" << std::endl;
    std::cout << "Window Width: " << args.WINDOW_WIDTH << std::endl;
    std::cout << "Window Height: " << args.WINDOW_HEIGHT << std::endl;
    std::cout << "Cell Size: " << args.CELL_SIZE << std::endl;
    std::cout << "Number of Threads: " << args.number_of_threads << std::endl;
    std::cout << "Type of Processing: " << args.type_of_processing << std::endl;

    int GRID_WIDTH = args.WINDOW_WIDTH / args.CELL_SIZE;
    int GRID_HEIGHT = args.WINDOW_HEIGHT / args.CELL_SIZE;

    std::vector<char> h_currentGrid(GRID_WIDTH * GRID_HEIGHT, 0);
    std::vector<char> h_newGrid(GRID_WIDTH * GRID_HEIGHT, 0);
    seedRandomGrid(h_currentGrid, GRID_WIDTH, GRID_HEIGHT);

    sf::RenderWindow window(sf::VideoMode(args.WINDOW_WIDTH, args.WINDOW_HEIGHT), "CUDA Game of Life");

    sf::Clock clock;
    unsigned long count = 0;
    sf::Time runTime = sf::Time::Zero;

    while (window.isOpen()) 
    {
        sf::Event event;
        while (window.pollEvent(event)) 
        {
            if (event.type == sf::Event::Closed) 
            {
                window.close();
            }
            if(event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) 
            {
                window.close();
            }
        }

        // Run CUDA processing
        if (args.type_of_processing == "NORMAL") 
        {
            normalMemoryAllocation(h_currentGrid.data(), h_newGrid.data(), GRID_WIDTH, GRID_HEIGHT, args.number_of_threads);
        } 
        else if (args.type_of_processing == "PINNED") 
        {
            pinnedMemoryAllocation(h_currentGrid.data(), h_newGrid.data(), GRID_WIDTH, GRID_HEIGHT, args.number_of_threads);
        } 
        else if (args.type_of_processing == "MANAGED") 
        {
            managedMemoryAllocation(h_currentGrid.data(), h_newGrid.data(), GRID_WIDTH, GRID_HEIGHT, args.number_of_threads);
        }

        runTime += clock.restart();
        if (++count % 100 == 0) 
        {
            std::cout << "100 generations took " << runTime.asMicroseconds() << " microseconds with " 
                      << args.number_of_threads << " threads using " << args.type_of_processing << " memory.\n";
            runTime = sf::Time::Zero;
        }

        // Render the grid
        window.clear();
        for (int x = 0; x < GRID_WIDTH; ++x) 
        {
            for (int y = 0; y < GRID_HEIGHT; ++y) 
            {
                if (h_newGrid[y * GRID_WIDTH + x]) 
                {
                    sf::RectangleShape cell(sf::Vector2f(args.CELL_SIZE, args.CELL_SIZE));
                    cell.setPosition(x * args.CELL_SIZE, y * args.CELL_SIZE);
                    cell.setFillColor(sf::Color::White);
                    window.draw(cell);
                }
            }
        }
        window.display();

        // Swap grids for next generation
        std::swap(h_currentGrid, h_newGrid);
    }

    return 0;
}