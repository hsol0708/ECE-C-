/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Nov 8,2024
Description: Lab 4: Cuda-Based John Conway’s Game of Life
What is the purpose of this file?
Command Line Argument File with reference to lecture 924
*/

// Include important C++ libraries here
#include "commandline.h"
#include <iostream>
#include <string>

//argc is number of arguments
Arguments commandLineArguments(int argc,char* argv[])
{
	Arguments args;

	args.WINDOW_WIDTH = 800;
    args.WINDOW_HEIGHT = 600;
    args.CELL_SIZE = 5;
    args.number_of_threads = 32;
    args.type_of_processing = "NORMAL";

	//Loop through command-line arguments
	for (int i=1; i < argc; ++i)
	{
		std::string arg = argv[i];

		if (arg == "-n" && (i + 1 < argc))
		{
			args.number_of_threads = std::stoi(argv[++i]);
			if (args.number_of_threads % 32 != 0)
			{
				std::cerr << " --number of threads (-n) must be a multiple of 32. Resetting to default n=32" << std::endl;
				args.number_of_threads = 32;
				exit(1);
			}
		}
		else if (arg == "-c" && (i + 1 < argc))
		{
			args.CELL_SIZE = std::stoi(argv[++i]);
			if (args.CELL_SIZE < 1)
			{
				std::cerr << " --number of threads (-c) must be greather than or equal to 1. Resetting to default c=5" << std::endl;
				args.CELL_SIZE = 5;
				exit(1);
			}
		}
		else if (arg == "-x" && (i + 1 < argc))
		{
			args.WINDOW_WIDTH = std::stoi(argv[++i]);
		}
		else if (arg == "-y" && (i + 1 < argc))
		{
			args.WINDOW_HEIGHT = std::stoi(argv[++i]);
		}
		else if (arg == "-t" && (i + 1 < argc)) {
            args.type_of_processing = argv[++i];
            std::string availabletypes[3] = { "NORMAL", "PINNED", "MANAGED" };
            bool isValidType = false;
            for (const std::string& str : availabletypes) {
                if (args.type_of_processing == str) {
                    isValidType = true;
                    break;
                }
            }
            if (!isValidType) {
                std::cerr << "--type of processing (-t) must be NORMAL, PINNED, or MANAGED. Resetting to default type NORMAL." << std::endl;
                args.type_of_processing = "NORMAL";
            }
        }
		else
		{
			std::cerr << "You might input an invalid command line. Please follow example ./cuda_sfml_app -c 5 -x 800 -y 600 -t NORMAL " << arg << std::endl;
			exit(1);
		}
	}


	return args; 
}
