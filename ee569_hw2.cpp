//  ee569_hw2.cpp
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
# include "ee569_hw2.hpp"
# include <stdio.h>
# include <iostream>
# include <fstream>
# include <sstream>
# include <stdlib.h>
# include <math.h>
# include "opencv2/imgcodecs.hpp"
# include "opencv2/highgui.hpp"
# include "opencv2/imgproc.hpp"
# include <opencv2/ximgproc.hpp>
# include "opencv2/core/utility.hpp"

using namespace std;

// Dynamic Allocating Array for Storing Image
// Return: unsigned char *** pnt
unsigned char *** ee569_hw2_sol::allocate_Image(int height, int width, int channel){
    unsigned char *** pnt = new unsigned char **[height];
    for(int i = 0; i < height; i++){
        pnt[i] = new unsigned char *[width];
        for(int j = 0; j < width; j ++){
            pnt[i][j] = new unsigned char [channel];
        }
    }
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            for(int k = 0; k < channel; k ++){
                pnt[i][j][k] = 0;
            }
        }
    }
    return pnt;
}

float *** ee569_hw2_sol::allocate_Image_float(int height, int width, int channel){
    float *** pnt = new float **[height];
    for(int i = 0; i < height; i++){
        pnt[i] = new float *[width];
        for(int j = 0; j < width; j ++){
            pnt[i][j] = new float [channel];
        }
    }
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            for(int k = 0; k < channel; k ++){
                pnt[i][j][k] = 0.;
            }
        }
    }
    return pnt;
}

bool *** ee569_hw2_sol::allocate_Image_bool(int height, int width, int channel){
    bool *** pnt = new bool **[height];
    for(int i = 0; i < height; i++){
        pnt[i] = new bool *[width];
        for(int j = 0; j < width; j ++){
            pnt[i][j] = new bool [channel];
        }
    }
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            for(int k = 0; k < channel; k ++){
                pnt[i][j][k] = false;
            }
        }
    }
    return pnt;
}

// Reading .raw images
unsigned char *** ee569_hw2_sol::input_raw(string filename, int height, int width, int channels, int padding, string extra_arg, bool zero_padding){
    unsigned char *** image_buff = allocate_Image(height + 2 * padding, width + 2 * padding, channels);
    // Read Image
    FILE * pFile;
    pFile = fopen((filename + extra_arg +".raw").c_str(), "rb");
    if (pFile == NULL){
        cout << "Cannot open file: " << filename <<endl;
    }
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fread(image_buff[i + padding][j + padding], sizeof(unsigned char), channels, pFile);
        }
    }
    // Fill in the values for reflection padding
    if(!zero_padding){
        for(int k = 0; k < channels; k ++){
            for(int i = padding; i < height + padding; i ++){
                for(int p = 0; p < padding; p ++){
                    image_buff[i][p][k] = image_buff[i][2 * padding - p][k];
                    image_buff[i][width + 2 * padding - 1 - p][k] = image_buff[i][width - 1 + p][k];
                }
            }
            for(int j = 0; j < width + 2 * padding; j ++){
                for(int p = 0; p < padding; p ++){
                    image_buff[p][j][k] = image_buff[2 * padding - p][j][k];
                    image_buff[height + 2 * padding - 1 - p][j][k] = image_buff[height - 1 + p][j][k];                
                }
            }
        }
    }
    cout << "Raw Image:" << filename << extra_arg << " Read Succeedded" << endl;
    return image_buff;
}

unsigned char *** ee569_hw2_sol::input_pgm(string filename){
    /* 
        Read pgm image, Code reference
        We are not varifying the headers for input images
        https://stackoverflow.com/questions/8126815/how-to-read-in-data-from-a-pgm-file-in-c
    */
    ifstream infile(filename + ".pgm");
    string buff;
    int height, width;
    getline(infile,buff);
    // Get Size
    infile >> buff;  width = stoi(buff); infile >> buff;  height = stoi(buff);
    getline(infile,buff); getline(infile,buff);
    unsigned char *** returnPnt = allocate_Image(height, width, 1);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            infile >> buff;
            returnPnt[i][j][0] = (unsigned char) stoi(buff);
        }
    }
    infile.close();
    return returnPnt;
}

unsigned char *** ee569_hw2_sol::input_ppm(string filename){
    /* 
        Read pgm image, Code reference
        We are not varifying the headers for input images
        https://stackoverflow.com/questions/8126815/how-to-read-in-data-from-a-pgm-file-in-c
    */
    ifstream infile(filename + ".ppm");
    string buff;
    int height, width;
    getline(infile,buff);
    // Get Size
    infile >> buff;  width = stoi(buff); infile >> buff;  height = stoi(buff);
    getline(infile,buff); getline(infile,buff);
    unsigned char *** returnPnt = allocate_Image(height, width, 3);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            for(int k = 0; k < 3; k ++){
                infile >> buff;
                returnPnt[i][j][k] = (unsigned char) stoi(buff);
            }
        }
    }
    infile.close();
    return returnPnt;
}

void ee569_hw2_sol::output_pgm(string filename, unsigned char *** image, int height, int width, int channel){
    if(print_pgmMessage)      cout << "Outputing PGM File for " << filename;
    ofstream pgmFile(filename + ".pgm");
    // Header 
    pgmFile << "P2" << endl << width << " " << height << endl << 255 << endl;
    // If gray image
    if (channel == 1){
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                pgmFile << (int) image[i][j][0] << " ";
            }
        }
    }
    if (channel == 3){
        double * weighted_sum;
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                weighted_sum = new double; // Need to check if allocation is 0
                * weighted_sum += MixUp_R * image[i][j][0];
                * weighted_sum += MixUp_G * image[i][j][1];
                * weighted_sum += MixUp_B * image[i][j][2];
                pgmFile << (int) * weighted_sum << " ";
            }
        }
        delete weighted_sum;
    }
    pgmFile.close();
    if(print_pgmMessage)      cout << ".......Finished" << endl;
}

void ee569_hw2_sol::output_ppm(string filename, unsigned char *** image, int height, int width, int channel){
    if(print_pgmMessage)      cout << "Outputing PPM File for " << filename;
    ofstream pgmFile(filename + ".ppm");
    pgmFile << "P3" << endl << width << " " << height << endl << 255 << endl;
    // If it's 24-bit, 3 channels RGB
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            pgmFile << (int) image[i][j][0] << " ";
            pgmFile << (int) image[i][j][1] << " ";
            pgmFile << (int) image[i][j][2] << endl;
        }
    }
    pgmFile.close();
    if(print_pgmMessage)      cout << ".......Finished" << endl;
}

void ee569_hw2_sol::output_raw(string filename, unsigned char *** image, int height, int width, int channel){
    if(print_pgmMessage)      cout << "Outputing raw File for " << filename;
    FILE * file;
    file = fopen((filename + ".raw").c_str(), "wb");
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fwrite(image[i][j], sizeof(unsigned char), channel, file);
        }
    }
    fclose(file);
    if(print_pgmMessage)      cout << ".......Finished" << endl;
}

unsigned char *** ee569_hw2_sol::rbg_to_gray(unsigned char *** image_rgb, int height, int width){
    unsigned char *** image_gray = allocate_Image(height, width, 1);
    double buff;
    for (int i = 0; i < height; i ++){
        for (int j = 0; j < width; j ++){
            buff = 0;
            buff += MixUp_R * (double) image_rgb[i][j][0];
            buff += MixUp_G * (double) image_rgb[i][j][1];
            buff += MixUp_B * (double) image_rgb[i][j][2];
            image_gray[i][j][0] = (unsigned char) buff;
        }
    }
    return image_gray;
}

bool *** ee569_hw2_sol::correspondPixel(unsigned char *** A, unsigned char *** B, int height, int width, int neighborSize){
    // Corespondence generated A -> B
    unsigned char pixel;
    int upleft_i, upleft_j, downright_i, downright_j;
    bool pixel_result, *** ret_mask = allocate_Image_bool(height, width, 1);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            pixel = A[i][j][0];
            if(!pixel)  continue;
            pixel_result = false;
            upleft_i = max(i - neighborSize, 0); upleft_j = max(j - neighborSize, 0);
            downright_i = min(i + neighborSize, height - 1); downright_j = min(j + neighborSize, width - 1);
            for(int temp_i = upleft_i; temp_i <= downright_i; temp_i ++){
                for(int temp_j = upleft_j; temp_j <= downright_j; temp_j ++){
                    pixel_result |= (pixel == B[temp_i][temp_j][0]);
                    if(pixel_result)    break;
                }
            }
            ret_mask[i][j][0] = pixel_result;
        }
    }
    return ret_mask;
}

double ee569_hw2_sol::f1_score_cal(unsigned char *** pred_image, string filename, int height, int width, int TYPE = 0){
    double ret_precision = 0, ret_recall = 0;
    int targetType;
    // Pixel-to-Pixel wise correspodence & F1 calculation
    if(filename == "Gallery")       targetType = 0;
    else if(filename == "Dogs")     targetType = 1;
    if(TYPE == 0){
        double tp, fp, tn, fn;
        for(int k = 0; k < 5; k ++){
            tp = 0.; fp = 0.; tn = 0.; fn = 0.;
            for(int i = 0; i < height; i ++){
                for(int j = 0; j < width; j ++){
                    if(pred_image[i][j][0] == 255 && Edge_truth_pool[targetType][k][i][j][0] == 255)       tp += 1.;
                    else if(pred_image[i][j][0] == 0 && Edge_truth_pool[targetType][k][i][j][0] == 255)    fn += 1.;
                    else if(pred_image[i][j][0] == 255 && Edge_truth_pool[targetType][k][i][j][0] == 0)    fp += 1.;
                    else if(pred_image[i][j][0] == 0 && Edge_truth_pool[targetType][k][i][j][0] == 0)      tn += 1.;
                }
            }
            if((tp + fp + tn + fn) != height * width)   throw "The result doesn't match !!!!";
            // cout << tp << ";" << fp << ";" << fn << ";" << tn << ";" << tp / (tp + fp) << ";" << tp / (tp + fn) << endl;
            ret_precision += 0.2 * tp / (tp + fp);
            ret_recall += 0.2 * tp / (tp + fn);
        }
    }
    // F1 Score calculation Method by StructuredForest/edgesEvalImg.m
    // Maximum Distance is 0.0075 * Image Diagnal, as defaulted by Structured Edge Eval, in this case = 4.337
    // Each pixel has 57 potential neighbors and we need 9 * 9 mask (Is not square)
    // Here we are implementing 7 * 7 mask with similar number of neighbors & squared mask
    // F1_cal supporting arbitary distance will be supported later :(
    else if(TYPE == 1){
        // For each Ground Truth Image
        float sumP = 0., sumR = 0.;
        bool *** EdgeMask, *** GTMask;
        bool *** matchE = allocate_Image_bool(height, width, 1);
        float *** matchG = allocate_Image_float(height, width, 1);
        for(int k = 0; k < 5; k ++){
            EdgeMask = correspondPixel(pred_image, Edge_truth_pool[targetType][k], height, width, 2);
            GTMask = correspondPixel(Edge_truth_pool[targetType][k], pred_image, height, width, 2);
            for(int i = 0; i < height; i ++){
                for(int j = 0; j < width; j++){
                    matchE[i][j][0] |= EdgeMask[i][j][0];
                    matchG[i][j][0] += (float) GTMask[i][j][0];
                    sumR += ((float) Edge_truth_pool[targetType][k][i][j][0] / 255.);
                    sumP += ((float) pred_image[i][j][0] / 255.);
                }
            }
        }
        // We are using the same variable name as Reference MATLAB code
        sumP /= 5.;
        float cntR = 0., cntP = 0.;
        for(int i = 0; i < height; i ++){
            for(int j = 0; j < width; j ++){
                cntR += matchG[i][j][0];
                cntP += (float) matchE[i][j][0];
            }
        }
        cout << cntP << " " << sumP << " " << cntR << " " << sumR << endl;
        ret_precision = cntP / sumP;
        ret_recall = cntR / sumR;
    }
    // Return Mean F1 score
    double f1 = 2 * ret_precision * ret_recall / (ret_precision + ret_recall);
    if(print_f1)    cout << "Mean F1 Score  " << f1 << endl;
    return f1;
}

float ** ee569_hw2_sol::kernel_init(string kernel_type, int kernel_height, int kernel_width){
    float ** kernel = new float *[kernel_height];
    for(int i = 0; i < kernel_height; i++){
        kernel[i] = new float [kernel_width];
    }
    for(int i = 0; i < kernel_height; i ++){
        for(int j = 0; j < kernel_width; j ++){
            kernel[i][j] = 0.;
        }
    }
    if (kernel_type == "Sobel_hor"){
        if (kernel_height != 3 || kernel_width != 3)
            throw "Input size of Sobel Kernels has to be 3 !";
        kernel[0][0] = 1./4; kernel[0][2] = -1./4;
        kernel[1][0] = 1./2; kernel[1][2] = -1./2;
        kernel[2][0] = 1./4; kernel[2][2] = -1./4;
    }
    else if (kernel_type == "Sobel_ver"){
        if (kernel_height != 3 || kernel_width != 3)
            throw "Input size of Sobel Kernels has to be 3 !";
        kernel[0][0] = -1./4; kernel[0][1] = -1./2; kernel[0][2] = -1./4;
        kernel[2][0] = 1./4;  kernel[2][1] = 1./2;  kernel[2][2] = 1./4; 
    }
    else if (kernel_type == "Dithering"){
        // Generate Dithering Index Kernel Iteratively
        if (kernel_height != kernel_width)
            throw "Dithering Index Matrix Has to be Square !";
        else if(((int)log2((double)kernel_height) % 1) != 0)
            throw "Dithering Index Matrix Has to be Power of 2 !";

        int currSize = 1;
        float ** currKernel, ** nextKernel;
        currKernel = new float *[currSize]; currKernel[0] = new float[currSize]; currKernel[0][0] = 0;
        while(currSize <= (kernel_height / 2)){
            nextKernel = new float *[2 * currSize];
            for(int i = 0; i < 2 * currSize; i++){
                nextKernel[i] = new float [2 * currSize];
            }
            for(int i = 0; i < currSize; i ++){
                for(int j = 0; j < currSize; j ++){
                    nextKernel[i][j]                           = currKernel[i][j] * 4 + 1;
                    nextKernel[i + currSize][j]                = currKernel[i][j] * 4 + 3;
                    nextKernel[i][j + currSize]                = currKernel[i][j] * 4 + 2;
                    nextKernel[i + currSize][j + currSize]     = currKernel[i][j] * 4 + 0;
                }
            }
            delete[] currKernel;
            currKernel = nextKernel;
            currSize *= 2;
        }
        return currKernel;
    }
    else if (kernel_type == "Floyd_ori"){
        if (kernel_height != 3 || kernel_width != 3)
            throw "Floyd-Steinberg has to be 3 by 3";
        kernel[0][0] = 0.; kernel[0][1] = 0.; kernel[0][2] = 0.; 
        kernel[1][0] = 0.; kernel[1][1] = 0.; kernel[1][2] = 7. / 16; 
        kernel[2][0] = 3. / 16; kernel[2][1] = 5. / 16; kernel[2][2] = 1. / 16; 
    }
    else if (kernel_type == "Floyd_mirror"){
        if (kernel_height != 3 || kernel_width != 3)
            throw "Floyd-Steinberg has to be 3 by 3";
        kernel[0][0] = 0.; kernel[0][1] = 0.; kernel[0][2] = 0.; 
        kernel[1][0] = 7. / 16; kernel[1][1] = 0.; kernel[1][2] = 0.; 
        kernel[2][0] = 1. / 16; kernel[2][1] = 5. / 16; kernel[2][2] = 3. / 16; 
    }
    else if (kernel_type == "JJN_ori"){
        if (kernel_height != 5 || kernel_width != 5)
            throw "Floyd-Steinberg has to be 5 by 5";
        kernel[2][3] = 7./48; kernel[2][4] = 5./48;
        kernel[3][0] = 3./48; kernel[3][1] = 5./48; kernel[3][2] = 7./48; kernel[3][3] = 5./48; kernel[3][4] = 3./48;
        kernel[4][0] = 1./48; kernel[4][1] = 3./48; kernel[4][2] = 5./48; kernel[4][3] = 5./48; kernel[4][4] = 1./48;
    }
    else if (kernel_type == "JJN_mirror"){
        if (kernel_height != 5 || kernel_width != 5)
            throw "Floyd-Steinberg has to by 5 by 5";
        kernel[2][1] = 7./48; kernel[2][0] = 5./48;
        kernel[3][0] = 3./48; kernel[3][1] = 5./48; kernel[3][2] = 7./48; kernel[3][3] = 5./48; kernel[3][4] = 3./48;
        kernel[4][0] = 1./48; kernel[4][1] = 3./48; kernel[4][2] = 5./48; kernel[4][3] = 5./48; kernel[4][4] = 1./48;
    }
    else if (kernel_type == "Stucki_ori"){
        if (kernel_height != 5 || kernel_width != 5)
            throw "Floyd-Steinberg has to by 5 by 5";
        kernel[2][3] = 8./42; kernel[2][4] = 4./42;
        kernel[3][0] = 2./42; kernel[3][1] = 4./42; kernel[3][2] = 8./42; kernel[3][3] = 4./42; kernel[3][4] = 2./42;
        kernel[4][0] = 1./42; kernel[4][1] = 2./42; kernel[4][2] = 4./42; kernel[4][3] = 2./42; kernel[4][4] = 1./42;
    }
    else if (kernel_type == "Stucki_mirror"){
        if (kernel_height != 5 || kernel_width != 5)
            throw "Floyd-Steinberg has to by 5 by 5";
        kernel[2][1] = 8./42; kernel[2][0] = 4./42;
        kernel[3][0] = 2./42; kernel[3][1] = 4./42; kernel[3][2] = 8./42; kernel[3][3] = 4./42; kernel[3][4] = 2./42;
        kernel[4][0] = 1./42; kernel[4][1] = 2./42; kernel[4][2] = 4./42; kernel[4][3] = 2./42; kernel[4][4] = 1./42;
    }
    return kernel;
}

float ee569_hw2_sol::fixed_kernel_CONV(float ** kernel, int kernel_size, unsigned char *** image, int upperleft_i, int upperleft_j, int k){
    // This conv operation uses upperleft point for dimension alignment, to be compatible with even size kernels
    // We assume this kernel is square
    float result = 0;
    for(int i = 0; i < kernel_size; i ++){
        for(int j = 0; j < kernel_size; j ++){
            result += kernel[i][j] * image[upperleft_i + i][upperleft_j + j][k];
        }
    }
    return result;
}

void ee569_hw2_sol::Sobel_edge_detector(string filename, int height, int width, int channel, bool toy_normal = true){
    string path;
    if(toy_normal)      path = "./Sobel_" + filename + "/Toy_thresholding/";
    else                path = "./Sobel_" + filename + "/Improved_thresholding/";
    unsigned char *** Image_rgb = input_raw(filename, height, width, channel, 0, "", true);
    unsigned char *** Image_ori = rbg_to_gray(Image_rgb, height, width);
    output_ppm(filename + "_rgb", Image_rgb, height, width, 1);
    output_pgm(filename + "_ori", Image_ori, height, width, 1);
    delete[] Image_rgb;
    // Gradients are not restricted to 0 ~ 255, we need float
    float *** grad_hor = allocate_Image_float(height, width, 1);
    float *** grad_ver = allocate_Image_float(height, width, 1);
    unsigned char *** image_hor = allocate_Image(height, width, 1);
    unsigned char *** image_ver = allocate_Image(height, width, 1);
    float ** kernel_hor = kernel_init("Sobel_hor", 3, 3);
    float ** kernel_ver = kernel_init("Sobel_ver", 3, 3);    
    float curMin_hor = 256, curMax_hor = -256, buff_hor;
    float curMin_ver = 256, curMax_ver = -256, buff_ver;
    // Horizontal Edges
    for (int i = 0; i < height - 2; i ++){
        for (int j = 0; j < width - 2; j ++){
            buff_hor = abs(fixed_kernel_CONV(kernel_hor, 3, Image_ori, i, j, 0));
            buff_ver = abs(fixed_kernel_CONV(kernel_ver, 3, Image_ori, i, j, 0));
            curMin_hor = min(curMin_hor, buff_hor); curMax_hor = max(curMax_hor, buff_hor);
            curMin_ver = min(curMin_ver, buff_ver); curMax_ver = max(curMax_ver, buff_ver);
            grad_hor[i + 1][j + 1][0] = buff_hor;
            grad_ver[i + 1][j + 1][0] = buff_ver;
        }
    }
    // Normalization, and type casting
    float bias_hor = curMin_hor, ratio_hor = 255. / (curMax_hor - curMin_hor);
    float bias_ver = curMin_ver, ratio_ver = 255. / (curMax_ver - curMin_ver);
    for (int i = 0; i < height; i ++){
        for (int j = 0; j < width; j ++){
            image_hor[i][j][0] = (unsigned char) ((grad_hor[i][j][0] - bias_hor) * ratio_hor);
            image_ver[i][j][0] = (unsigned char) ((grad_ver[i][j][0] - bias_ver) * ratio_ver);
        }
    }
    output_pgm(filename + "_hor", image_hor, height, width, 1);
    output_pgm(filename + "_ver", image_ver, height, width, 1);
    // Compute gradient magnitude
    float *** grad_mag = allocate_Image_float(height, width, 1);
    unsigned char *** image_mag = allocate_Image(height, width, 1);
    float curMin = 256, curMax = -256, buff;
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            buff = sqrt(pow(grad_hor[i][j][0], 2) + pow(grad_ver[i][j][0], 2));
            curMin = min(curMin, buff); curMax = max(curMax, buff);
            grad_mag[i][j][0] = buff;
        }
    }
    // Note: Standard normalization is not good
    // Include statistical features of the image
    float bias, ratio;
    if(toy_normal){bias = curMin; ratio = 255. / (curMax - curMin);}
    else{
        if(filename == "Gallery"){bias = 10; ratio = 255. / (50 - 10);}
        else if(filename == "Dogs"){bias = 20; ratio = 255. / (80 - 20);}
    }
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            buff = (grad_mag[i][j][0] - bias) * ratio;
            if(buff >= 255)              image_mag[i][j][0] = 255;
            else if(buff <= 0)           image_mag[i][j][0] = 0;
            else                         image_mag[i][j][0] = (unsigned char) buff;
        }
    }
    output_pgm(filename + "_mag", image_mag, height, width, 1);
    // Cropping & Thresholding
    double bestF1 = 0, currF1;  string best_Setting;
    unsigned char *** Image_buff = allocate_Image(height, width, 1);
    for(int T = 0; T <= 256; T += 1){
        for(int i = 0; i < height; i ++){
            for(int j = 0; j < width; j ++){
                if(image_mag[i][j][0] >= T)     Image_buff[i][j][0] = 255;
                else                            Image_buff[i][j][0] = 0;
            }
        }
        currF1 = f1_score_cal(Image_buff, filename, height, width, 1);
        if(bestF1 < currF1){
            bestF1 = currF1; best_Setting = to_string(T);
        }
        if(print_pgmppmImage) output_pgm(path + filename + "_crop" + to_string(T), Image_buff, height, width, 1);
    }
    cout << "Best F1 Score:" << bestF1 << "  Best Setting:" << best_Setting << endl;
    //Print the best image
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            if(image_mag[i][j][0] >= stoi(best_Setting))     Image_buff[i][j][0] = 255;
            else                                             Image_buff[i][j][0] = 0;
        }
    }
    output_pgm("./GroundTruth/" + filename + "_Sobel", Image_buff, height, width, 1);
    // delete[] Image_rgb;  delete[] Image_ori; delete[] kernel_hor; delete[] kernel_ver;
    // delete[] image_hor;  delete[] image_ver; delete[] grad_hor;   delete[] grad_ver;
    // delete[] Image_buff; delete[] image_mag; delete[] grad_mag;
}

void ee569_hw2_sol::Canny_edge_detector(string filename, int height, int width, int channel){
    /*
        Code Reference: OpenCV Official Library
        https://docs.opencv.org/trunk/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    */
    string path = "./Canny_" + filename + "/";
    unsigned char *** Image_rgb = input_raw(filename, height, width, channel, 0, "", true);
    unsigned char *** Image_ori = rbg_to_gray(Image_rgb, height, width);
    output_pgm(filename + "_ori", Image_ori, height, width, 1);
    delete[] Image_rgb; delete[] Image_ori;
    cv::Mat Image_gray = cv::imread(filename + "_ori.pgm", cv::IMREAD_GRAYSCALE), Image_afterOp;
    // cv::blur(Image_gray, Image_gray, cv::Size(3, 3));
    unsigned char *** Image_buff = allocate_Image(height, width, 1);
    double bestF1 = 0, currF1;  string best_Setting;
    for(int low_T = 0; low_T <= 300; low_T += 10){
        for(int high_T = low_T + 10; high_T <= 500; high_T += 10){
            cv::Canny(Image_gray, Image_afterOp, low_T, high_T, 3, true);
            for(int i = 0; i < height; i ++){
                for(int j = 0; j < width; j ++){
                    Image_buff[i][j][0] = Image_afterOp.at<unsigned char>(i, j, 0);
                }
            }
            currF1 = f1_score_cal(Image_buff, filename, height, width, 1);
            if(bestF1 < currF1){
                bestF1 = currF1; best_Setting = to_string(low_T) + "|" + to_string(high_T);
            }
            if(print_pgmppmImage) output_pgm(path + filename + "_" + to_string(low_T) + "|" + to_string(high_T), Image_buff, height, width, 1);
        }
    }
    cout << "Best F1 Score:" << bestF1 << "  Best Setting:" << best_Setting << endl;
    // Print the best image
    int low_T = stoi(best_Setting.substr(0, best_Setting.find("|")));
    int high_T = stoi(best_Setting.substr(best_Setting.find("|") + 1, best_Setting.back()));
    cv::Canny(Image_gray, Image_afterOp, low_T, high_T, 3, true);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            Image_buff[i][j][0] = Image_afterOp.at<unsigned char>(i, j, 0);
        }
    }
    output_pgm("./GroundTruth/" + filename + "_Canny", Image_buff, height, width, 1);
    delete[] Image_buff;
}

void ee569_hw2_sol::Structured_edge_detector(string filename, int height, int width, int channel, bool img_show = false){
    /*
        Code Reference: OpenCV Official Library & Sample Source Code
        https://docs.opencv.org/3.4.9/d0/da5/tutorial_ximgproc_prediction.html
    */
    string path = "./SE_" + filename + "/";
    unsigned char *** Image_rgb = input_raw(filename, height, width, channel, 0, "", true);
    output_ppm(filename + "_rgb", Image_rgb, height, width, 1);
    delete[] Image_rgb;
    cv::Mat Image = cv::imread(filename + "_rgb.ppm", cv::IMREAD_COLOR);
    string modelFilename = "model.yml";
    Image.convertTo(Image, cv::DataType<float>::type, 1 / 255.0);
    cv::Mat edges(Image.size(), Image.type());
    // Read Pre-trained model
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar = cv::ximgproc::createStructuredEdgeDetection(modelFilename);
    pDollar -> detectEdges(Image, edges);
    // computes orientation from edge map & Suppress Edges
    cv::Mat orientation_map, edge_nms;
    pDollar -> computeOrientation(edges, orientation_map);
    pDollar -> edgesNms(edges, orientation_map, edge_nms, 2, 0, 1, true);
    // Output the edges 
    // Visually Best Threshold: 0.19 for Dogs , 0.17 for Gallery
    unsigned char *** Image_afterOp = allocate_Image(height, width, 1);
    // Print the direct output of SE algorithm
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j ++){
            Image_afterOp[i][j][0] = (unsigned char) (255 * (edge_nms.at<float>(i, j, 0)));
        }
    }
    output_pgm(filename + "_nms", Image_afterOp, height, width, 1);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j ++){
            Image_afterOp[i][j][0] = (unsigned char) (255 * (edges.at<float>(i, j, 0)));
        }
    }
    output_pgm(filename + "_edge", Image_afterOp, height, width, 1);
    
    double bestF1 = 0, currF1;  string best_Setting;
    for(float T = 0.00; T <= 1; T += 0.005){
        for(int i = 0; i < height; i ++){
            for(int j = 0; j < width; j ++){
                Image_afterOp[i][j][0] = (unsigned char) (255 * (edge_nms.at<float>(i, j, 0) >= T));
            }
        }
        currF1 = f1_score_cal(Image_afterOp, filename, height, width, 1);
        if(bestF1 < currF1){
            bestF1 = currF1; best_Setting = to_string(T);
        }
        if(print_pgmppmImage) output_pgm(path + filename + "|" + to_string(T), Image_afterOp, height, width, 1);
        if(img_show){
            cv::imshow("edges", edges);
            cv::imshow("edges nms", edge_nms);
            cv::waitKey(0);
        }
    }
    // Print the best image
    float T = stof(best_Setting);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            Image_afterOp[i][j][0] = (unsigned char) (255 * (edge_nms.at<float>(i, j, 0) >= T));
        }
    }
    output_pgm("./GroundTruth/" + filename + "_SE", Image_afterOp, height, width, 1);
    cout << "Best F1 Score:" << bestF1 << "  Best Setting:" << best_Setting << endl;
}

void ee569_hw2_sol::best_setting_show(string filename, int height, int width, int channel){
    string path = "./GroundTruth/";
    cout << filename << endl << "Sobel     " << "Canny     " << "SE" << endl;
    f1_score_cal(input_pgm(path + filename + "_Sobel"), filename, height, width, 1);
    f1_score_cal(input_pgm(path + filename + "_Canny"), filename, height, width, 1);
    f1_score_cal(input_pgm(path + filename + "_SE"), filename, height, width, 1);
    cout << endl;
}

void ee569_hw2_sol::Dithering_fixed_T(string filename, int height, int width, int channel){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, 1, 0, "", true);
    unsigned char *** Image_afterOp = allocate_Image(height, width, channel);
    unsigned char T = 128;
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            if(Image[i][j][0] >= T)     Image_afterOp[i][j][0] = 255;
            else if(Image[i][j][0] < T) Image_afterOp[i][j][0] = 0;
        }
    }
    output_pgm(path + filename + "FixedT|" + to_string(T), Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Dithering_random_T(string filename, int height, int width, int channel){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, 1, 0, "", true);
    output_pgm(path + filename + "_ori", Image, height, width, channel);
    unsigned char *** Image_afterOp = allocate_Image(height, width, channel);
    unsigned char T;
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            T = rand() % 256;
            if(Image[i][j][0] >= T)     Image_afterOp[i][j][0] = 255;
            else if(Image[i][j][0] < T) Image_afterOp[i][j][0] = 0;
        }
    }
    output_pgm(path + filename + "UniRandT|" + to_string(T), Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Dithering_Mx(string filename, int height, int width, int channel, int kernel_dim){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, 1, 0, "", true);
    unsigned char *** Image_afterOp = allocate_Image(height, width, 1);
    float ** kernel = kernel_init("Dithering", kernel_dim, kernel_dim);
    output_pgm(path + filename + "_ori", Image, height, width, channel);
    double T;
    for(int i = 0; i < height; i ++ ){
        for(int j = 0; j < width; j ++){
            T = (kernel[i % kernel_dim][j % kernel_dim] + 0.5) * 255. / pow(kernel_dim, 2);
            if(Image[i][j][0] >= T)     Image_afterOp[i][j][0] = 255;
            else if(Image[i][j][0] < T) Image_afterOp[i][j][0] = 0;
        }
    }
    output_pgm(path + filename + "Matrix|" + to_string(kernel_dim), Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Error_diffuse_step(float *** Image, float ** kernel, int kernel_size,
                                       int upperleft_i, int upperleft_j, int channel, float * error){
    for(int k = 0; k < channel; k ++){
        for(int i = 0; i < kernel_size; i ++){
            for(int j = 0; j < kernel_size; j ++){
                if (kernel[i][j] == 0)  continue;
                Image[upperleft_i + i][upperleft_j + j][k] += kernel[i][j] * error[k];
                if(Image[upperleft_i + i][upperleft_j + j][k] > 1.)       {Image[upperleft_i + i][upperleft_j + j][k] = 1.;}
                else if (Image[upperleft_i + i][upperleft_j + j][k] < 0.) {Image[upperleft_i + i][upperleft_j + j][k] = 0.;}
            }
        }
    }
}

unsigned char *** ee569_hw2_sol::Error_diffuse_DO(float *** kernel, int padding, float *** scr_Image, float T,
                                                  int height, int width, int channel, string TYPE = "Naive"){
    unsigned char *** ret_Image = allocate_Image(height, width, channel);
    unsigned char * quantization_result = new unsigned char [channel];
    float * err = new float [channel];
    int col = padding, flag = 1;
    for(int i = padding; i < height + padding; i ++){
        while((padding - 1) < col && col < (width + padding)){
            quantization_result = fetch_closest(TYPE, scr_Image[i][col], channel, T);
            for(int k = 0; k < channel; k ++){
                err[k] = scr_Image[i][col][k] - (float) quantization_result[k];
                ret_Image[i - padding][col - padding][k] = 255 * quantization_result[k];
            }
            Error_diffuse_step(scr_Image, kernel[(flag == -1)], 1 + 2 * padding, i - padding, col - padding, channel, err);
            col += flag;
        }
        flag *= -1;
        col += flag;
    }
    return ret_Image;
}

void ee569_hw2_sol::Error_diffusion_Floyd(string filename, int height, int width, int channel){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, channel, 1, "", true);
    float *** Image_0_1 = allocate_Image_float(height + 2, width + 2, 1);
    float *** kernel = new float **[2];
    kernel[0] = kernel_init("Floyd_ori", 3, 3);
    kernel[1] = kernel_init("Floyd_mirror", 3, 3);
    // Normalize the Image
    for(int i = 0; i < height + 2; i ++){
        for(int j = 0; j < width + 2; j ++){
            Image_0_1[i][j][0] = (float) Image[i][j][0] / 255.;
        }
    }
    unsigned char *** Image_afterOp = Error_diffuse_DO(kernel, 1, Image_0_1, 0.5, height, width, channel);
    output_pgm(path + filename + "_Floyd", Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Error_diffusion_JJN(string filename, int height, int width, int channel){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, channel, 2, "", true);
    float *** Image_0_1 = allocate_Image_float(height + 4, width + 4, 1);
    float *** kernel = new float **[2];
    kernel[0] = kernel_init("JJN_ori", 5, 5);
    kernel[1] = kernel_init("JJN_mirror", 5, 5);
    // Normalize the Image
    for(int i = 0; i < height + 4; i ++){
        for(int j = 0; j < width + 4; j ++){
            Image_0_1[i][j][0] = (float) Image[i][j][0] / 255.;
        }
    }
    unsigned char *** Image_afterOp = Error_diffuse_DO(kernel, 2, Image_0_1, 0.5, height, width, channel);
    output_pgm(path + filename + "_JJN", Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Error_diffusion_Stucki(string filename, int height, int width, int channel){
    string path = "./HalfTone/";
    unsigned char *** Image = input_raw(filename, height, width, channel, 2, "", true);
    float *** Image_0_1 = allocate_Image_float(height + 4, width + 4, 1);
    float *** kernel = new float **[2];
    kernel[0] = kernel_init("Stucki_ori", 5, 5);
    kernel[1] = kernel_init("Stucki_mirror", 5, 5);
    // Normalize the Image
    for(int i = 0; i < height + 4; i ++){
        for(int j = 0; j < width + 4; j ++){
            Image_0_1[i][j][0] = (float) Image[i][j][0] / 255.;
        }
    }
    unsigned char *** Image_afterOp = Error_diffuse_DO(kernel, 2, Image_0_1, 0.5, height, width, channel);
    output_pgm(path + filename + "_Stucki", Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::Error_diffusion_NaiveColored(string filename, string TYPE, int height, int width, int channel){
    string path = "./HalfTone/";
    int kernel_dim;
    if(TYPE == "Floyd")     kernel_dim = 1;
    else                    kernel_dim = 2;
    unsigned char *** Image = input_raw(filename, height, width, channel, kernel_dim, "", true);
    float *** Image_0_1 = allocate_Image_float(height + 2 * kernel_dim, width + 2 * kernel_dim, 3);
    float *** kernel = new float **[2];
    kernel[0] = kernel_init(TYPE + "_ori", 2 * kernel_dim + 1, 2 * kernel_dim + 1);
    kernel[1] = kernel_init(TYPE + "_mirror", 2 * kernel_dim + 1, 2 * kernel_dim + 1);
    // Normalize the Image
    for(int k = 0; k < channel; k ++){
        for(int i = 0; i < height + 2 * kernel_dim; i ++){
            for(int j = 0; j < width + 2 * kernel_dim; j ++){
                Image_0_1[i][j][k] = (float) Image[i][j][k] / 255.;
            }
        }
    }
    unsigned char *** Image_afterOp = Error_diffuse_DO(kernel, kernel_dim, Image_0_1, 0.5, height, width, channel, "Naive");
    output_ppm(path + filename + "_(Naive)" + TYPE, Image_afterOp, height, width, channel);
}

unsigned char ee569_hw2_sol::fetch_quadruples(float * RGB_at_ij){
    // This locating algorithm is suggested by the MBVQ paper
    float R = RGB_at_ij[0], G = RGB_at_ij[1], B = RGB_at_ij[2];
    if((R + G) > 1.){
        if((G + B) > 1.){
            if((R + G + B) > 2.)     {return CMYW;}
            else                     {return MYGC;}
        }
        else                         {return RGMY;}
    }
    else{
        if(!((G + B) > 1.)){
            if(!((R + G + B) > 1.))  {return KRGB;}
            else                     {return RGBM;}
        }
        else                         {return CMGB;}
    }
}

unsigned char * ee569_hw2_sol::fetch_closest(string TYPE, float * RGB_at_ij, int channel, float T){
    // According to the paper, all Lp yields the same result, use L2
    unsigned char * closestColor = new unsigned char [channel];
    if(TYPE == "Naive"){
        for(int k = 0; k < channel; k ++){
            closestColor[k] = (unsigned char) (RGB_at_ij[k] >= T);
        }
    }
    else if(TYPE == "MBVQ"){
        unsigned char quadruples = fetch_quadruples(RGB_at_ij);
        // bitPnt points to the current color we're pointing to
        int bitPnt = 0, bestPnt = 0;
        float best = 10., temp;
        while(bitPnt < 8){
            if(quadruples % 2 != 0){
                temp = 0.;
                for(int k = 0; k < channel; k ++){
                    temp += pow(RGB_at_ij[k] - (float) mapping[bitPnt][k], 2);
                }
                if(temp < best){
                    bestPnt = bitPnt;
                    best = temp;
                }
            }
            quadruples >>= 1;
            bitPnt += 1;
        }
        for(int k = 0; k < channel; k ++){
            closestColor[k] = mapping[bestPnt][k];
        }
    }
    return closestColor;
}

void ee569_hw2_sol::Error_diffusion_MBVQ(string filename, string TYPE, int height, int width, int channel){
    string path = "./HalfTone/";
    int kernel_dim;
    if(TYPE == "Floyd")     kernel_dim = 1;
    else                    kernel_dim = 2;
    unsigned char *** Image = input_raw(filename, height, width, channel, kernel_dim, "", true);
    output_ppm(path + filename + "_ori", Image, height, width, 3);
    float *** Image_0_1 = allocate_Image_float(height + 2 * kernel_dim, width + 2 * kernel_dim, 3);
    float *** kernel = new float **[2];
    kernel[0] = kernel_init(TYPE + "_ori", 2 * kernel_dim + 1, 2 * kernel_dim + 1);
    kernel[1] = kernel_init(TYPE + "_mirror", 2 * kernel_dim + 1, 2 * kernel_dim + 1);
    // Normalize the Image
    for(int k = 0; k < channel; k ++){
        for(int i = 0; i < height + 2 * kernel_dim; i ++){
            for(int j = 0; j < width + 2 * kernel_dim; j ++){
                Image_0_1[i][j][k] = (float) Image[i][j][k] / 255.;
            }
        }
    }
    unsigned char *** Image_afterOp = Error_diffuse_DO(kernel, kernel_dim, Image_0_1, 0.5, height, width, channel, "MBVQ");
    output_ppm(path + filename + "_(MBVQ)" + TYPE, Image_afterOp, height, width, channel);
}

void ee569_hw2_sol::edge_error_visualization(string filename, string Target, int height, int width, int channel){
    string path = "./GroundTruth/";
    // First Synthesize all the ground truth labels
    unsigned char *** weighted_truth = allocate_Image(height, width, 1);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            for(int k = 0; k < 5; k ++){
                weighted_truth[i][j][0] += ((int) Edge_truth_pool[filename == "Dogs"][k][i][j][0]) / 5;
            }
        }
    }
    output_pgm(path + filename + "GT_weighted", weighted_truth, height, width, 1);
    unsigned char *** pred = input_pgm(path + filename + "_" + Target);
    unsigned char *** ret_image = allocate_Image(height, width, 3);
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j ++){
            if(weighted_truth[i][j][0] != 0 && pred[i][j][0] == 255)       ret_image[i][j][1] = weighted_truth[i][j][0];
            else if(weighted_truth[i][j][0] != 0 && pred[i][j][0] == 0)    ret_image[i][j][0] = weighted_truth[i][j][0];
            else if(weighted_truth[i][j][0] == 0 && pred[i][j][0] == 255)  ret_image[i][j][2] = 255;
            else if(weighted_truth[i][j][0] == 0 && pred[i][j][0] == 0){
                ret_image[i][j][0] = 255; ret_image[i][j][1] = 255; ret_image[i][j][2] = 255; 
            }

        }
    }
    output_ppm(path + filename + "GT_" + Target + "_Visualize", ret_image, height, width, 3);
    delete[] weighted_truth; delete[] pred; delete[] ret_image;
}

void ee569_hw2_sol::EDGE_DETECTOR_VISUALIZE_ALL(){
    edge_error_visualization("Gallery", "Sobel", 321, 481, 3);
    edge_error_visualization("Gallery", "Canny", 321, 481, 3);
    edge_error_visualization("Gallery", "SE", 321, 481, 3);
    edge_error_visualization("Dogs", "Sobel", 321, 481, 3);
    edge_error_visualization("Dogs", "Canny", 321, 481, 3);
    edge_error_visualization("Dogs", "SE", 321, 481, 3);
}

void ee569_hw2_sol::EDGE_DETECTOR_DO_ALL(bool f1, bool pgmMessage, bool pgmppmImage){
    print_f1 = f1;
    print_pgmMessage = pgmMessage;
    print_pgmppmImage = pgmppmImage;
    // Sobel_edge_detector("Gallery", 321, 481, 3, false);
    // Sobel_edge_detector("Dogs",    321, 481, 3, false);
    // Sobel_edge_detector("Gallery", 321, 481, 3, true);
    // Sobel_edge_detector("Dogs",    321, 481, 3, true);
    // Canny_edge_detector("Gallery", 321, 481, 3);
    // Canny_edge_detector("Dogs",    321, 481, 3);
    Structured_edge_detector("Gallery", 321, 481, 3, false);
    Structured_edge_detector("Dogs",    321, 481, 3, false);
    best_setting_show("Gallery", 321, 481, 3);
    best_setting_show("Dogs", 321, 481, 3);
}

void ee569_hw2_sol::HALF_TUNE_DO_ALL(){
    Dithering_fixed_T("LightHouse", 500, 750, 1);
    Dithering_random_T("LightHouse", 500, 750, 1);
    Dithering_Mx("LightHouse", 500, 750, 1, 2);
    Dithering_Mx("LightHouse", 500, 750, 1, 4);
    Dithering_Mx("LightHouse", 500, 750, 1, 8);
    Dithering_Mx("LightHouse", 500, 750, 1, 32);
    Dithering_Mx("LightHouse", 500, 750, 1, 64);
    Dithering_Mx("LightHouse", 500, 750, 1, 128);
    Dithering_Mx("LightHouse", 500, 750, 1, 256);
    Error_diffusion_Floyd("LightHouse", 500, 750, 1);
    Error_diffusion_JJN("LightHouse", 500, 750, 1);
    Error_diffusion_Stucki("LightHouse", 500, 750, 1);
    Error_diffusion_NaiveColored("Rose", "Floyd", 480, 640, 3);
    Error_diffusion_NaiveColored("Rose", "JJN", 480, 640, 3);
    Error_diffusion_NaiveColored("Rose", "Stucki", 480, 640, 3);
    Error_diffusion_MBVQ("Rose", "Floyd", 480, 640, 3);
    Error_diffusion_MBVQ("Rose", "JJN", 480, 640, 3);
    Error_diffusion_MBVQ("Rose", "Stucki", 480, 640, 3);
}

void ee569_hw2_sol::f1_test(){
    // This function serves as a test unit for f1_cal
    cout << "---------------THIS_IS_A_TEST---------------------" << endl;
    cout << "Toy 3 by 3 Manually entered Matrix" << endl;
    unsigned char pred[3][3] = {{0, 255, 0},
                                {0, 255, 0},
                                {255, 0, 0}};
    unsigned char true_[3][3] = {{0, 255, 255},
                                 {255, 0, 0},
                                 {0, 0, 255}};


    double tp, fp, tn, fn, ret_precision = 0, ret_recall = 0;
    tp = 0.; fp = 0.; tn = 0.; fn = 0.;
    for(int i = 0; i < 3; i ++){
        for(int j = 0; j < 3; j ++){
            if(pred[i][j] == 255 && true_[i][j] == 255)       tp += 1.;
            else if(pred[i][j] == 0 && true_[i][j] == 255)    fn += 1.;
            else if(pred[i][j] == 255 && true_[i][j] == 0)    fp += 1.;
            else if(pred[i][j] == 0 && true_[i][j] == 0)      tn += 1.;
        }
    }
    if((tp + fp + tn + fn) != 3 * 3)   throw "The result doesn't match !!!!";
    // cout << tp << ";" << fp << ";" << fn << ";" << tn << ";" << tp / (tp + fp) << ";" << tp / (tp + fn) << endl;
    ret_precision = tp / (tp + fp);
    ret_recall = tp / (tp + fn);
    // Return Mean F1 score
    double f1 = 2 * ret_precision * ret_recall / (ret_precision + ret_recall);
    cout << "Mean F1 Score  " << f1 << endl;
    cout << "This is a test for bit operations" << endl;
    unsigned char quadruples = MYGC;
    int bitPnt = 0;
    while(bitPnt < 8){
        cout << quadruples % 2 << endl;
        bitPnt += 1;
        quadruples >>= 1;
    }


    cout << "--------------------TEST_ENDS---------------------" << endl;
}

void ee569_hw2_sol::pxm_to_raw_DO_ALL(){
    unsigned char *** Image_buff;
    Image_buff = input_pgm("Dogs_Canny");
    output_raw("Dogs_Canny", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("Dogs_SE");
    output_raw("Dogs_SE", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("Dogs_Sobel");
    output_raw("Dogs_Sobel", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("Gallery_Canny");
    output_raw("Gallery_Canny", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("Gallery_SE");
    output_raw("Gallery_SE", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("Gallery_Sobel");
    output_raw("Gallery_Sobel", Image_buff, 321, 481, 1);
    Image_buff = input_pgm("LightHouse_Floyd");
    output_raw("LightHouse_Floyd", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouse_JJN");
    output_raw("LightHouse_JJN", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouse_Stucki");
    output_raw("LightHouse_Stucki", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouseFixedT|128");
    output_raw("LightHouseFixedT|128", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouseMatrix|2");
    output_raw("LightHouseMatrix|2", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouseMatrix|8");
    output_raw("LightHouseMatrix|8", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouseMatrix|32");
    output_raw("LightHouseMatrix|32", Image_buff, 500, 750, 1);
    Image_buff = input_pgm("LightHouseUniRandT");
    output_raw("LightHouseUniRandT", Image_buff, 500, 750, 1); 
    Image_buff = input_ppm("Rose_(MBVQ)Floyd");
    output_raw("Rose_(MBVQ)Floyd", Image_buff, 480, 640, 3); 
    Image_buff = input_ppm("Rose_(MBVQ)JJN");
    output_raw("Rose_(MBVQ)JJN", Image_buff, 480, 640, 3); 
    Image_buff = input_ppm("Rose_(MBVQ)Stucki");
    output_raw("Rose_(MBVQ)Stucki", Image_buff, 480, 640, 3); 
    Image_buff = input_ppm("Rose_(Naive)Floyd");
    output_raw("Rose_(Naive)Floyd", Image_buff, 480, 640, 3); 
    Image_buff = input_ppm("Rose_(Naive)JJN");
    output_raw("Rose_(Naive)JJN", Image_buff, 480, 640, 3); 
    Image_buff = input_ppm("Rose_(Naive)Stucki");
    output_raw("Rose_(Naive)Stucki", Image_buff, 480, 640, 3); 
}