/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Nov 25,2024
Description:  Lab 6: Using OpenMPI to Estimate the Value of a Definite Integral using the Monte Carlo method
What is the purpose of this file?
Program for estimation of value of definite integral using Monte Carlo method
*/

#include "mpi.h"
#include <iostream>
#include <cstring>
#include <cmath>

#define MAIN 0

// Assigning random seeds
void srandom(unsigned seed);

// Setting the integral case 
double integral(int& rank, int& number_Tasks, int& number_Samples, int& Integral_Problem_Number)
{
    long random(void);
    double start = (double)rank / number_Tasks;
    double timestep = 1.0 / number_Tasks;
    double x, y, sum = 0;
	
    unsigned int c;  
    
    if (sizeof(c) != 4) 
    {
        printf("Error, the constant must be 4-bytes\n");
        exit(1);
    }

    c = 2 << (31 - 1);
	
    if (Integral_Problem_Number == 1)
    {
        for (int i = 0; i < number_Samples; i++)
        {
            x = start + ((double)random() / c) * timestep;
            y = x * x;
            sum += y;
        }

    }

	// Integral_Problem_Number ==2
	else
    {
        for (int i = 0; i < number_Samples; i++)
        {
            x = start + ((double)random() / c) * timestep;
            y = -1 * x * x;
            y = std::exp(y);
            sum += y;
        }
    }
    return sum;
}



int main(int argc, char *argv[])
{
    double local_Sum, global_Sum, result;
	
    int taskID, number_Tasks, i, Integral_Problem_Number, number_Samples, totalSamples, rc;

	// Parsing the input argument from user
    for (i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-P") == 0)
        {
			// Assign which integral to use
            Integral_Problem_Number = std::stoi(argv[i + 1]);
            // Theres only problem number 1 and 2
            if (Integral_Problem_Number != 1 && Integral_Problem_Number != 2)
            {
                std::cout << "Invalid problem" << std::endl;
                return 1;
            }
        } 
		else if (strcmp(argv[i], "-N") == 0)
        {
			// Assign the total number of samples
            totalSamples = std::stoi(argv[i + 1]);
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &number_Tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskID);

    srandom(taskID);
	
	// split the samples based of the number of task.
    number_Samples = (int)(totalSamples) / number_Tasks;
	
	// Master task check
    if (taskID == MAIN)
    {
        int m_samples = number_Samples + totalSamples % number_Tasks;
		local_Sum = integral(taskID, number_Tasks, m_samples, Integral_Problem_Number);
    } 
	else
    {
        local_Sum = integral(taskID, number_Tasks, number_Samples, Integral_Problem_Number);
    }

	// Combine all local data for global sum
    rc = MPI_Reduce(&local_Sum, &global_Sum, 1, MPI_DOUBLE, MPI_SUM, MAIN, MPI_COMM_WORLD);

    if (taskID == MAIN)
    {
		result = global_Sum / totalSamples;
        std::cout << "The estimate for integral is " << result << std::endl;
        std::cout << "Bye!" << std::endl;
    }
}