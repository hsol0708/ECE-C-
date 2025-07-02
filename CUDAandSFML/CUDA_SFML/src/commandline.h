/*
Author: Fan Han Hoon
Class:  ECE6122 (Fall 2024)
Last Date Modified: Sep 30,2024
Description: Lab 1: Retro Centipede Arcade Game
What is the purpose of this file?
CommandLine parser file
*/
#pragma once
#include <string>

struct Arguments
{
	int number_of_threads = 32;
	int CELL_SIZE = 5;
	int WINDOW_WIDTH = 800;
	int WINDOW_HEIGHT = 600;
	std::string type_of_processing = "NORMAL";

};

Arguments commandLineArguments(int argc, char* argv[]);