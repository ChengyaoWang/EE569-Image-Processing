# EE569 Homework Assignment #1
# Date: Jan 28, 2020
# Name: Chengyao Wang
# ID: 6961-5998016
# email: chengyao@usc.edu
#
# Software: Visual Studio Code (Only for editing)
# Operating System: macOS Catalina 10.15.3 BETA
# Compiler: clang++, using command line tools
========================================================================
	THIS IS THE README FOR HOMEWORK1
========================================================================

Steps:
1. All the files need for duplicating the results:
	main.cpp: main function
	ee569_hw1.cpp:	all the source code for Homework1
	ee569_hw1.hpp:	Header file of *.cpp
	All the .raw images
	hw1_sol.command: Shell script for automated generation
	plotting.py:	Python script reading C++ written output.csv, and plotting.

2. Open source code used:
	OpenCV official Library:
		This is only for BM3D, but may not compile & run other functions if missing.
	Matplotlib:
		Python Library for plotting

3. Double click hw1_sol.command file, and will automatically replicate results.
	First line of .command file is used to re-direct to the current directory, change if errors with this respect happen
	Compiler Command: clang++ $(pkg-config --cflags --libs opencv4) -std=c++11 main.cpp ee569_hw1.cpp -o main
	Arguments passing into the main are organized as:
	executable, name of the original image, width & height, channels, demosaicing, histogram manipulation, denoising,
	Note: parameter tuning process is turned off.

4. For simplify the checking process of outputted images:
	*.pgm, *.ppm format are used to inspect the intermediate results, they are supported by the MacOS system.
	*.raw files will also be generated.
   
5. 	Please contact my email if any error happens.
	Sadly, I cannot foresee every one of them :(
	Also, OpenCV doesn't support outputting *.raw image :( thus here's *pgm on its behalf.

/////////////////////////////////////////////////////////////////////////////
