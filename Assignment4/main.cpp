//
//  main.cpp
//  opencv_playground
//
//  Created by Chengyao Wang on 1/14/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//

# include "ee569_hw4.hpp"
# include <stdio.h>

using namespace std;




int main(int argc, char *argv[]){
    cout << "-------------------PROGRAM STARTS----------------------" << endl;
    ee569_hw4_sol object;
    // object.READ_ALL_RAW();
    // object.Feature_Visualization();


    // object.TextureClassification_Unsupervised();
    // object.TextureClassification_Supervised_SVM();
    // object.TextureClassification_Supervised_RF();


    object.TextureClassification_Segmentation();


    
    cout << "-----------------FUNCTION TERMINATES-------------------" << endl;
    return 0;
}