# EE569 Homework Assignment #2
# Date: Feb 15, 2020
# Name: Chengyao Wang
# ID: 6961-5998-16
# email: chengyao@usc.edu
#
# Software: Visual Studio Code (Only for editing)
# Operating System: macOS Catalina 10.15.4 BETA
# Compiler: clang++, using command line tools
========================================================================
	THIS IS THE README FOR HOMEWORK2
========================================================================

Steps:
1. All the files need for duplicating the results:
	main.cpp: 		main function
	ee569_hw2.cpp:		all the source code for Homework2
	ee569_hw2.hpp:		Header file of *.cpp
	All the .raw images
	hw2_sol.command: 	Shell script
	model.yml:		Pre-trained model for SE
	Ground Truth Images
	Correct directory tree
		./Canny_Dogs
		./Canny_Gallery
		./GroundTruth
		./HalfTone
		./SE_Dogs
		./SE_Gallery
		./Sobel_Dogs/Improved_thresholding
		./Sobel_Dogs/Toy_thresholding
		./Sobel_Gallery/Improved_thresholding
		./Sobel_Gallery/Toy_thresholding

2. Open source code used:
	OpenCV official Library:
		This is only for SE / Canny, but may not compile & run other functions if missing.

3. Double click hw2_sol.command file, and will automatically replicate results.
	First line of .command file is used to re-direct to the current directory, change if errors with this respect happen
	Compiler Command: clang++ $(pkg-config --cflags --libs opencv4) -std=c++11 main.cpp ee569_hw2.cpp -o main
	Arguments passing into the main are organized as:
	Note: 	parameter tuning process is always turned on, outputting images during tuning is turned off. 
		else the tuning process will generate 1+GB images.
		Parameter tuning process may lead to program crash due to memory insufficiencies. 

4. For simplify the checking process of outputted images:
	*.pgm, *.ppm format are used to inspect the intermediate results, they are supported by the MacOS system.
	*.raw files will also be generated.
   
5. 	Please contact my email if any error happens.
	Sadly, I cannot foresee every one of them :(
	Also, OpenCV doesn't support outputting *.raw image :( thus here's *pgm on its behalf.
/////////////////////////////////////////////////////////////////////////////
