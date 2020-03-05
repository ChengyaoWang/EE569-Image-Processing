//
//  main.cpp
//  opencv_playground
//
//  Created by Chengyao Wang on 1/14/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//

# include "ee569_hw3.hpp"

# include <stdio.h>
# include <iostream>
# include <fstream>
# include <stdlib.h>


using namespace std;


int main(int argc, char *argv[]){
    cout << "Program Starts" << endl;

    ee569_hw3_sol object;

    // object.READ_ALL_RAW();

    object.Geometric_Warping("../hedwig", 512, 512, 3);
    object.Geometric_Warping("../raccoon", 512, 512, 3);
    object.Geometric_Warping("../bb8", 512, 512, 3);

    vector<string> filenameList = {"../left", "../middle", "../right"};
    vector<int>    heightList = {720, 720, 720};
    vector<int>    widthList = {480, 480, 480};

    object.Image_Stitching(filenameList, heightList, widthList, 3, 1, 4);


    object.MorphologicalProcess_Basic(0, 100, 10);
    object.MorphologicalProcess_CountStars("../stars", 480, 640, 1);
    object.MorphologicalProcess_PCBanalysis("../PCB", 239, 372, 1);
    object.MorphologicalProcess_DefeatDetection("../GearTooth", 250, 250, 1);

    // object.pxm_TO_RAW_ALL();

    cout << "-----------------FUNCTION TERMINATES-------------------" << endl;
    return 0;
}