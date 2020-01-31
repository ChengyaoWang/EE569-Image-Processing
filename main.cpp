//
//  main.cpp
//  opencv_playground
//
//  Created by Chengyao Wang on 1/14/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//

# include "ee569_hw1.hpp"

# include <stdio.h>
# include <iostream>
# include <fstream>
# include <stdlib.h>



using std::cin;
using std::cout;
using std::endl;


int main(int argc, char *argv[]){
    cout << "Program Starts" << endl;
    ee569_hw1_sol hw1_sol;
    hw1_sol.read_image_raw(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    if(atoi(argv[5])){
        hw1_sol.demosaicing_bilinear();
        hw1_sol.demosaicing_MHC();
    }
    if(atoi(argv[6])){
        hw1_sol.enhancement_transfer_function();
        hw1_sol.enhancement_cumulative_p();
    }
    if(atoi(argv[7])){
        bool stage_select[3] = {false, false, false};
        hw1_sol.parameter_tuning(stage_select);
        hw1_sol.denoising_uniform(5, true);
        hw1_sol.denoising_gaussian(5, 0.88, true);
        hw1_sol.denoising_bilateral(5, 2.625, 81, true);
        hw1_sol.denoising_NLM(16.2, 16.6, true, 19);
        hw1_sol.denoising_BM3D();
    }
    cout << "---------------------------------" << endl;
    return 0;
}
