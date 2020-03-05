//  ee569_hw1.cpp
//  opencv_playground
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
# include "ee569_hw1.hpp"
# include <stdio.h>
# include <iostream>
# include <fstream>
# include <stdlib.h>
# include <vector>
# include <random>
# include <math.h>
# include "opencv2/imgcodecs.hpp"
# include "opencv2/highgui.hpp"
# include "opencv2/xphoto/bm3d_image_denoising.hpp"

using namespace std;

// Dynamic Allocating Array for Storing Image
// Return: unsigned char ***pnt
unsigned char *** ee569_hw1_sol::allocate_Image(int height, int width, int channel){
    unsigned char *** pnt = new unsigned char **[height];
    for(int i = 0; i < height; i++){
        pnt[i] = new unsigned char *[width];
        for(int j = 0; j < width; j ++){
            pnt[i][j] = new unsigned char [channel];
        }
    }
    return pnt;
}

// Read images in raw format
void ee569_hw1_sol::read_image_raw(string filename, int width, int height, int channels){
    Image_width = width;
    Image_height = height;
    Image_channels = channels;
    Image_filename = filename;
    if(Image_filename == "Corn_noisy"){
        Image_filename = "Corn";
    }
    // Directly consider the padding which will be later used, padding = 2
    Image = allocate_Image(Image_height + 4, Image_width + 4, Image_channels);
    // Read Image
    FILE * pFile;
    pFile = fopen((filename + ".raw").c_str(), "rb");
    if (pFile == NULL){
        cout << "Cannot open file: " << filename <<endl;
    }
    for(int i = 0; i < Image_height; i++){
        for(int j = 0;j < Image_width; j++){
            fread(Image[i + 2][j + 2], sizeof(unsigned char), Image_channels, pFile);
        }
    }
    // If inputing Corn image, also read noiseless image
    if(Image_filename == "Corn"){
        Image_noiseless = allocate_Image(Image_height, Image_width, Image_channels);
        pFile = fopen((Image_filename + "_gray.raw").c_str(), "rb");
        if (pFile == NULL){
            cout << "Cannot open file: " << "Corn_gray.raw" <<endl;
        }
        for(int i = 0; i < Image_height; i++){
            for(int j = 0;j < Image_width; j++){
                fread(Image_noiseless[i][j], sizeof(unsigned char), Image_channels, pFile);
            }
        }
        print_pgm(Image_filename + "_gray", Image_noiseless, Image_height, Image_width, Image_channels);
    }
    else if(Image_filename == "Dog"){
        Image_noiseless = allocate_Image(Image_height, Image_width, 3);
        pFile = fopen((Image_filename + "_ori.raw").c_str(), "rb");
        if (pFile == NULL){
            cout << "Cannot open file: " << "Dog_ori.raw" <<endl;
        }
        for(int i = 0; i < Image_height; i++){
            for(int j = 0;j < Image_width; j++){
                fread(Image_noiseless[i][j], sizeof(unsigned char), 3, pFile);
            }
        }
        print_ppm(Image_filename + "_ori", Image_noiseless, Image_height, Image_width, 3);
    }

    // Fill in the values for reflection padding
    // The order of filling: Columns -> Rows, But it can be shown that doesn't matter
    // Luckily, due to size of the images, edge cases don't exist
    for(int i = 0; i < Image_height + 4; i++){
        for(int k = 0; k < Image_channels; k++){
            Image[i][0][k] = Image[i][4][k];
            Image[i][1][k] = Image[i][3][k];
            Image[i][Image_width + 2][k] = Image[i][Image_width][k];
            Image[i][Image_width + 3][k] = Image[i][Image_width - 1][k];
        }
    }
    for(int j = 0; j < Image_width + 4; j++){
        for(int k = 0; k < Image_channels; k++){
            Image[0][j][k] = Image[4][j][k];
            Image[1][j][k] = Image[3][j][k];
            Image[Image_height + 2][j][k] = Image[Image_height][j][k];
            Image[Image_height + 3][j][k] = Image[Image_height - 1][j][k];
        }
    }
    // Output Its ppm file & pgm file
    print_pgm(Image_filename, Image, Image_height + 4, Image_width + 4, Image_channels);
    print_ppm(Image_filename, Image, Image_height + 4, Image_width + 4, Image_channels);
    cout << "PGM & PPM file outputed" << endl;
}

void ee569_hw1_sol::print_pgm(string target_filename, unsigned char *** target_image, int target_height, int target_width, int target_channel){
    cout << "Outputing PGM File for " << target_filename;
    ofstream pgmFile(target_filename + ".pgm"); // output image file we're creating
    pgmFile << "P2" << endl;
    pgmFile << target_width << " " << target_height << endl; // how many columns, how many rows
    pgmFile << 255 << endl << endl << endl; // largest pixel value we'll be outputting (below)
    // If it's 8-bit, one channel
    if (target_channel == 1){
        for(int i = 0; i < target_height; i++){
            for(int j = 0; j < target_width; j++){
                pgmFile << (int) target_image[i][j][0] << " ";
            }// next column
        }// next row
    }
    // If it's 24-bit, one channel.
    // Mixing with 1:1:1 ratio
    if (target_channel == 3){
        for(int i = 0; i < target_height; i++){
            for(int j = 0; j < target_width; j++){
                pgmFile << (target_image[i][j][0] + target_image[i][j][1] + target_image[i][j][2]) / 3 << " ";
            }// next column
            pgmFile << endl;
        }// next row
    }
    pgmFile.close();
    cout << ".......Finished" << endl;
}

void ee569_hw1_sol::print_ppm(string target_filename, unsigned char *** target_image, int target_height, int target_width, int target_channel){
    cout << "Outputing PPM File for " << target_filename;
    ofstream pgmFile(target_filename + ".ppm"); // output image file we're creating
    pgmFile << "P3" << endl;
    pgmFile << target_width << " " << target_height << endl; // how many columns, how many rows
    pgmFile << 255 << endl << endl; // largest pixel value we'll be outputting (below)
    // If it's only 8-bit, one channel
    if (target_channel == 1){
        for(int i = 0; i < target_height; i++){
            for(int j = 0; j < target_width; j++){
                // If green pixel
                if ((i + j) % 2 == 0){
                    pgmFile << 0 << " " << (int) target_image[i][j][0] << " " << 0 << endl;
                }
                // If Red pixel
                else if (j % 2 == 1){
                    pgmFile << (int) target_image[i][j][0] << " " << 0 << " " << 0 << endl;
                }
                // If Blue
                else{
                    pgmFile << 0 << " " << 0 << " " << (int) target_image[i][j][0] << endl;
                }
            }
        }
    }
    // If it's 24-bit, 3 channels RGB
    else if (target_channel == 3){
        for(int i = 0; i < target_height; i++){
            for(int j = 0; j < target_width; j++){
                pgmFile << (int) target_image[i][j][0] << " " << (int) target_image[i][j][1] << " " << (int) target_image[i][j][2] << endl;
            }
        }
    }
    pgmFile.close();
    cout << ".......Finished" << endl;
}

void ee569_hw1_sol::output_raw(string filename, unsigned char *** image, int channel){
    FILE * file;
    file = fopen((filename + ".raw").c_str(), "wb");

    for(int i = 0; i < Image_height; i++){
        for(int j = 0; j < Image_width; j++){
            fwrite(image[i][j], sizeof(unsigned char), channel, file);
        }
    }
    fclose(file);
    cout << "Succeeded" << endl;
}

// Currently only supports 2D kernels
float ** ee569_hw1_sol::kernel_init(string kernel_name, int size, string target_color = "R", string center_color = "R", double std_c = 0){
    float ** kernel = new float *[5];
    for(int i = 0; i < 5; i++){
        kernel[i] = new float [5];
    }
    // Ensure Initialization to zero
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            kernel[i][j] = 0.0;
        }
    }
    if(kernel_name == "bilinear" || kernel_name == "MHC"){
        if(center_color == target_color){
            kernel[2][2] = 1.;
            if(kernel_name == "MHC"){}
        }
        else if(target_color == "G_odd" || target_color == "G_even"){
            kernel[1][2] = 1./4;
            kernel[2][1] = 1./4;    kernel[2][3] = 1./4;
            kernel[3][2] = 1./4;
            if(kernel_name == "MHC"){
                kernel[0][2] = -MHC_alpha/4;
                kernel[2][0] = -MHC_alpha/4; kernel[2][2] = MHC_alpha;    kernel[2][4] = -MHC_alpha/4;
                kernel[4][2] = -MHC_alpha/4;
            }
        }
        else if((center_color == "R" && target_color == "B") || (center_color == "B" && target_color == "R")){
            kernel[1][1] = 1./4; kernel[1][3] = 1./4;
            kernel[3][1] = 1./4; kernel[3][3] = 1./4;
            if(kernel_name == "MHC"){
                kernel[0][2] = -MHC_gamma/4;
                kernel[2][0] = -MHC_gamma/4; kernel[2][2] = MHC_gamma;    kernel[2][4] = -MHC_gamma/4;
                kernel[4][2] = -MHC_gamma/4;
            }
        }
        else if((center_color == "G_odd" && target_color == "R") || (center_color == "G_even" && target_color == "B")){
            kernel[1][2] = 1./2;
            kernel[3][2] = 1./2;
            if(kernel_name == "MHC"){
                kernel[0][2] = -MHC_beta/5;
                kernel[1][1] = -MHC_beta/5; kernel[1][3] = -MHC_beta/5;
                kernel[2][0] = MHC_beta/10; kernel[2][2] = MHC_beta;    kernel[2][4] = MHC_beta/10;
                kernel[3][1] = -MHC_beta/5; kernel[3][3] = -MHC_beta/5;
                kernel[4][2] = -MHC_beta/5;
            }
        }
        else if((center_color == "G_odd" && target_color == "B") || (center_color == "G_even" && target_color == "R")){
            kernel[2][1] = 1./2; kernel[2][3] = 1./2;
            if(kernel_name == "MHC"){
                kernel[0][2] = MHC_beta/10;
                kernel[1][1] = -MHC_beta/5; kernel[1][3] = -MHC_beta/5;
                kernel[2][0] = -MHC_beta/5; kernel[2][2] = MHC_beta;    kernel[2][4] = -MHC_beta/5;
                kernel[3][1] = -MHC_beta/5; kernel[3][3] = -MHC_beta/5;
                kernel[4][2] = MHC_beta/10;
            }
        }
    }
    else if(kernel_name == "uniform"){
        for(int i = int(size == 3); i < 5 - int(size == 3); i++){
            for(int j = int(size == 3); j < 5 - int(size == 3); j++){
                kernel[i][j] = 1.0 / (size * size);
            }
        }
    }
    else if(kernel_name == "gaussian"){
        // Weight for Gaussian kernel is discrete valued
        double sum = 0;
        for(int i = int(size == 3); i < 5 - int(size == 3); i++){
            for(int j = int(size == 3); j < 5 - int(size == 3); j++){
                kernel[i][j] = exp( - metric_cal(abs(i - 2), abs(j - 2), "euclidean") / pow(std_c, 2));
                sum += kernel[i][j];
            }
        }
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < 5; j++){
                kernel[i][j] /= sum;
            }
        }
    }
    else if(kernel_name == "bilateral"){
        // Initialize space information part of the kernel
        for(int i = int(size == 3); i < 5 - int(size == 3); i++){
            for(int j = int(size == 3); j < 5 - int(size == 3); j++){
                kernel[i][j] = exp( - (pow(abs(i - 2), 2) + pow(abs(j - 2), 2)) / pow(std_c, 2));
            }
        }
    }
    else if(kernel_name == "NLM"){}
    return kernel;
}

double ee569_hw1_sol::metric_cal(double x_diff, double y_diff, string metric_type){
    double result = 0;
    if(metric_type == "euclidean"){
        result = sqrt(pow(x_diff, 2) + pow(y_diff, 2));
    }
    return result;
}

double ee569_hw1_sol::PSNR_cal(unsigned char *** raw_image, unsigned char *** denoised_image){
    double mse = 0;
    for(int i = 0; i < Image_height; i++){
        for(int j = 0; j < Image_width; j++){
            for(int k = 0; k < Image_channels; k++){
                mse += pow(raw_image[i][j][k] - denoised_image[i][j][k], 2);
            }
        }
    }
    mse /= (Image_height * Image_width * Image_channels);
    return 10 * log10(255.0 * 255.0 / mse);
}

unsigned char ee569_hw1_sol::conv_op_demoisaic(int center_i, int center_j, string kernel_type, string target_color){
    // Determine center color
    float result = 0;
    string center_color;
    if((center_i + center_j) % 2 == 0){
        if(center_i % 2 == 1)           center_color = "G_odd";
        else if(center_i % 2 == 0)      center_color = "G_even";
    }
    else if(center_j % 2 == 1)              center_color = "R";
    else                                    center_color = "B";
    float ** kernel = kernel_init(kernel_type, 5, target_color, center_color);
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            result += kernel[i][j] * Image[center_i + i][center_j + j][0];
        }
    }
    if(result < 0.0){
        result = 0;
    }
    else if(result > 255.0){
        result = 255;
    }
    return (unsigned char)(result);
}

unsigned char ee569_hw1_sol::conv_op_denoise_fixedKernel(int center_i, int center_j, int channel, float ** kernel){
    float result = 0;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            result += kernel[i][j] * Image[center_i + i][center_j + j][channel];
        }
    }
    return (unsigned char) (result);
}

unsigned char ee569_hw1_sol::conv_op_denoise_bilateral(int center_i, int center_j, int channel, float std_s, float ** kernel){
    float result = 0, weight_sum = 0;
    float I2 = Image[center_i + 2][center_j + 2][channel];
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j ++){
            float I1 = Image[center_i + i][center_j + j][channel];
            float intense_diff = exp( - pow(I1 - I2, 2) / pow(std_s, 2));
            result += kernel[i][j] * intense_diff * Image[center_i + i][center_j + j][channel];
            weight_sum += kernel[i][j] * intense_diff;
            //cout << kernel[i][j] << " " << kernel[i][j] * intense_diff << endl;
        }
        //cout << endl;
    }
    return (unsigned char) (result / weight_sum);
}

unsigned char ee569_hw1_sol::conv_op_denoise_NLM(int center_i, int center_j, int channel, int window_size, float h, float a){
    float result = 0, weight_sum = 0;
    // Determine the location of the large window we're sampling from, (A, B) -> (C, D)
    int A = max(0, center_i - window_size / 2);
    int B = max(0, center_j - window_size / 2);
    int C = min(Image_height, center_i + window_size / 2);
    int D = min(Image_width, center_j + window_size / 2);
    for(int i = A; i < C; i++){
        for(int j = B; j < D; j++){
            // For this certain region, calculate f(i, j, k, l)
            float weight = 0;
            for(int i_temp = 0; i_temp < 5; i_temp++){
                for(int j_temp = 0; j_temp < 5; j_temp++){
                    float g_value = exp( - (pow(i_temp - 2, 2) + pow(j_temp - 2, 2)) / (2 * pow(a, 2)));
                    weight += g_value * pow(Image[i + i_temp][j + j_temp][channel] -
                                            Image[center_i + i_temp][center_j + j_temp][channel], 2);
                }
            }
            weight = exp( - weight / pow(h, 2) / (sqrt(2 * M_PI) * a) );
            result += Image[i + 2][j + 2][channel] * weight;
            weight_sum += weight;
        }
    }
    return (unsigned char) (result / weight_sum);
}

void ee569_hw1_sol::demosaicing_bilinear(){
    // Channel No. are arranged as R -> G -> B
    // No padding for the output image
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 3);
    for(int i = 0; i < Image_height; i++){
        for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][0] = conv_op_demoisaic(i, j, "bilinear", "R");
                if(i % 2 == 1)          Image_afterOp[i][j][1] = conv_op_demoisaic(i, j, "bilinear", "G_odd");
                else if(i % 2 == 0)     Image_afterOp[i][j][1] = conv_op_demoisaic(i, j, "bilinear", "G_even");
                Image_afterOp[i][j][2] = conv_op_demoisaic(i, j, "bilinear", "B");
        }
    }
    // Print out the Image
    print_ppm("Bilinear_" + Image_filename, Image_afterOp, Image_height, Image_width, 3);
    output_raw("Bilinear_" + Image_filename, Image_afterOp, 3);
    cout << "Demosaicing by Bilinear Kernel Generated" << endl;
}

void ee569_hw1_sol::demosaicing_MHC(){
    // Channel No. are arranged as R -> G -> B
    // No padding for the output image
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 3);
    for(int i = 0; i < Image_height; i++){
        for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][0] = conv_op_demoisaic(i, j, "MHC", "R");
                if(i % 2 == 1)          Image_afterOp[i][j][1] = conv_op_demoisaic(i, j, "MHC", "G_odd");
                else if(i % 2 == 0)     Image_afterOp[i][j][1] = conv_op_demoisaic(i, j, "MHC", "G_even");
                Image_afterOp[i][j][2] = conv_op_demoisaic(i, j, "MHC", "B");
        }
    }
    // Print out the Image
    print_ppm("MHC_" + Image_filename, Image_afterOp, Image_height, Image_width, 3);
    output_raw("MHC_" + Image_filename, Image_afterOp, 3);
    cout << "Demosaicing by MHC Kernel Generated" << endl;
}

void ee569_hw1_sol::enhancement_transfer_function(){
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 3);
    ofstream csvFile("output.csv");
    for(int k = 0; k < 3; k++){
        // This is one channel, calculate the distribution
        float * mapping = new float[256];
        float total_pixel = Image_height * Image_width;
        for(int i = 0; i < 256; i++)        mapping[i] = 0;
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                mapping[Image[2 + i][2 + j][k]]++;
            }
        }
        // Print the Emperical PDF of pixel intensities each channel into csv file
        for(int i = 0; i < 256; i++){
            csvFile << (int) mapping[i] << ", ";
        }
        csvFile << endl;
        // Cumulation
        float sum_so_far = 0;
        for(int i = 0; i < 256; i++){
            sum_so_far += mapping[i];
            mapping[i] = sum_so_far * 255 / total_pixel;
        }
        // Map the pixel intensities to normal distribution
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][k] = (unsigned int) mapping[Image[2 + i][2 + j][k]];
            }
        }
        // Print the transfer function
        for(int i = 0; i < 256; i++){
            csvFile << (int) mapping[i] << ", ";
        }
        csvFile << endl;
        delete[] mapping;
    }
    csvFile.close();
    cout << "Enhancement by Transfer Function Generated" << endl;
    cout << "Transfer Function / PDF for Pixel Intensites outputed" << endl;
    print_ppm("transfer_" + Image_filename, Image_afterOp, Image_height, Image_width, 3);
    output_raw("transfer_" + Image_filename, Image_afterOp, 3);
}

void ee569_hw1_sol::enhancement_cumulative_p(){
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 3);
    // In this case, pixel_per_intensity = 875.0
    int pixel_per_intensity = Image_height * Image_width / 256;
    // Stack is used for shuffling
    vector<int *> * myStack = new vector<int *> [256];
    // Define pnt for iteratively allocation of memory
    int * pnt = new int[2];
    for(int k = 0; k < 3; k++){
        // No need to re-initializae myStack & pnt
        // myStack is empty after the previous excution
        // Store location information according to pixel intensities
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                pnt = new int[2];
                pnt[0] = i; pnt[1] = j;
                myStack[Image[2 + i][2 + j][k]].push_back(pnt);
            }
        }
        // Shuffle all the vectors;
        for(int i = 0; i < 256; i++){
            shuffle(myStack[i].begin(), myStack[i].end(), default_random_engine());
        }
        // Use double pointers to re-calibrate the pixel intensites
        int myStack_shift = 0;
        for(int cur_Intensity = 0; cur_Intensity < 256; cur_Intensity++){
            for(int i = 0; i < pixel_per_intensity; i++){
                while(myStack[myStack_shift].empty() == true)       myStack_shift += 1;
                pnt = myStack[myStack_shift].back();
                Image_afterOp[pnt[0]][pnt[1]][k] = (unsigned char)cur_Intensity;
                myStack[myStack_shift].pop_back();
            }
        }
    }
    cout << "Enhancement by Cumulative Probability Generated" << endl;
    print_ppm("Cumul_prob_" + Image_filename, Image_afterOp, Image_height, Image_width, 3);
    output_raw("Cumul_prob_" + Image_filename, Image_afterOp, 3);
}

double ee569_hw1_sol::denoising_uniform(int kernel_size, bool verbose){
    // Channel No. are arranged as R -> G -> B
    // No padding for the output image
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 1);
    float ** kernel = kernel_init("uniform", kernel_size);
    for(int k = 0; k < Image_channels; k++){
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][k] = conv_op_denoise_fixedKernel(i, j, k, kernel);
            }
        }
    }
    // Print out the Image
    if(verbose == true){
        print_pgm("Uniform" + to_string(kernel_size) + "_" + Image_filename, Image_afterOp, Image_height, Image_width, 1);
        cout << "Denoising by Uniform Kernel Size " << char(kernel_size) << " Generated" << endl;
        output_raw("Uniform" + to_string(kernel_size) + "_" + Image_filename, Image_afterOp, 1);
    }
    cout << "PSNR score:" << PSNR_cal(Image_noiseless, Image_afterOp) << endl;
    return PSNR_cal(Image_noiseless, Image_afterOp);
}

double ee569_hw1_sol::denoising_gaussian(int kernel_size, float std_a, bool verbose){
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 1);
    float ** kernel = kernel_init("gaussian", kernel_size, "R", "R", std_a);
    for(int k = 0; k < Image_channels; k++){
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][k] = conv_op_denoise_fixedKernel(i, j, k, kernel);
            }
        }
    }
    // Print out the Image
    if(verbose == true){
        print_pgm("Gaussian" + to_string(kernel_size) + "_" + Image_filename, Image_afterOp, Image_height, Image_width, 1);
        cout << "Denoising by Gaussian Kernel Size " << char(kernel_size) << " Generated" << endl;
        output_raw("Gaussian" + to_string(kernel_size) + "_" + Image_filename, Image_afterOp, 1);
    }
    cout << "PSNR score:" << PSNR_cal(Image_noiseless, Image_afterOp) << " With std_a = " << std_a << endl;
    return PSNR_cal(Image_noiseless, Image_afterOp);
}

double ee569_hw1_sol::denoising_bilateral(int kernel_size, float std_a, float std_s, bool verbose){
    // Channel No. are arranged as R -> G -> B
    // No padding for the output image
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 1);
    float ** kernel = kernel_init("bilateral", kernel_size, "R", "R", std_a);
    for(int k = 0; k < Image_channels; k++){
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][k] = conv_op_denoise_bilateral(i, j, k, std_s, kernel);
            }
        }
    }
    // Print out the Image
    if(verbose == true){
        print_pgm("Bilateral" + to_string(std_a) + "_" + to_string(std_s) + "_" + to_string(kernel_size) + "_" + Image_filename,
                  Image_afterOp, Image_height, Image_width, 1);
        output_raw("Bilateral_" + Image_filename, Image_afterOp, 1);
        cout << "Denoising by Bilateral Kernel Size " << char(kernel_size) << " Generated" << endl;
    }
    cout << "PSNR score:" << PSNR_cal(Image_noiseless, Image_afterOp);
    cout << " With std_a = " << std_a << " std_s = " << std_s << endl;
    return PSNR_cal(Image_noiseless, Image_afterOp);
}

double ee569_hw1_sol::denoising_NLM(float std_a, float std_h, bool verbose, int corpus_size = 21){
    // Determine the large window location (A, B) upper left -> (C, D) lower right
    unsigned char *** Image_afterOp = allocate_Image(Image_height, Image_width, 1);
    for(int k = 0; k < Image_channels; k++){
        for(int i = 0; i < Image_height; i++){
            for(int j = 0; j < Image_width; j++){
                Image_afterOp[i][j][k] = conv_op_denoise_NLM(i, j, k, corpus_size, std_h, std_a);
            }
        }
    }
    // Print out the Image
    if(verbose == true){
        print_pgm("NLM" + to_string(std_a) + "_" + to_string(std_h) + "_" + Image_filename,
                  Image_afterOp, Image_height, Image_width, 1);
        output_raw("NLM_" + Image_filename, Image_afterOp, 1);
        cout << "Denoising by NLM Kernel Size 5 Generated" << endl;
    }
    cout << "PSNR score:" << PSNR_cal(Image_noiseless, Image_afterOp);
    cout << " With std_a = " << std_a << " std_h = " << std_h << "Corpus_size" << corpus_size <<endl;
    return PSNR_cal(Image_noiseless, Image_afterOp);
}
 
// This function generates the parameter tuning process of HW1
void ee569_hw1_sol::parameter_tuning(bool stage_select[]){
    // Stage 1: Gaussian Kernel,  std_a : spacial information
    // Best Found: std_a = 0.88, with PSNR score:19.4265
    // Stage 2: Bilateral Kernel, std_a: spacial information
    //                            std_s: pixel intensity information
    // Best Found: PSNR score:19.5915 With std_a = 2.625 std_s = 77
    // Stage 3: NLM:              corpus size: 21
    //                            std_a: spacial information
    //                            std_h: pixel intensity information
    // Best Found: PSNR score: 19.7447 With (std_a, std_h) = 16.2, 16.6
    double cur_max = 0;
    if(stage_select[0]){
        for(int i = 58; i < 118; i++){
            float std_a = i / 100.0;
            denoising_gaussian(5, std_a, false);
        }
    }
    if(stage_select[1]){
        for(float i = 100; i < 500; i += 1){
            //for(float j = 72; j < 80; j += 0.1){
                cur_max = max(cur_max, denoising_bilateral(5, i / 100.0, 77 / 1.0, false));
            //}
        }
        cout << cur_max << endl;
    }
    if(stage_select[2]){
        for(float i = 16; i < 18; i += 0.2){
            for(float j = 16; j < 18; j += 0.2){
                cur_max = max(cur_max, denoising_NLM(i / 1.0, j / 1.0, false));
            }
        }
        cout << cur_max << endl;
    }   
}

void ee569_hw1_sol::denoising_BM3D(){
    // Reference https://www.ipol.im/pub/art/2012/l-bm3d/article.pdf
    // API used: https://docs.opencv.org/4.2.0/de/daa/group__xphoto.html#ga2fc5a9661c1338a823fb3290673a880d
    // Official Sample Usage of bm3dDenoising():
    //      https://github.com/opencv/opencv_contrib/blob/master/modules/xphoto/test/test_denoise_bm3d.cpp
    // Read pgm image -> read image using imread() -> do BM3D -> output image
    // This part is rather independent of the previous part of the homework, and is dependent on OpenCV C++ lib
    // This is the official code for implementation of cv::xphoto::bm3dDenoising()
    // TEST(xphoto_DenoisingBm3dGrayscale, regression_L2_8x8)
    // {
    //     Mat original = cv::imread(original_path, cv::IMREAD_GRAYSCALE);
    //     Mat expected = cv::imread(expected_path, cv::IMREAD_GRAYSCALE);
    //     Mat result;
    //     xphoto::bm3dDenoising(original, result, 10, 8, 16, 2500, -1, 8, 1, 0.0f, NORM_L2, xphoto::BM3D_STEP1);
    //     DUMP(result, expected_path + ".res.png");
    // }
    // @paras (default):
    //      float 	h = 1,
    //      int 	templateWindowSize = 4,
    //      int 	searchWindowSize = 16,
    //      int 	blockMatchingStep1 = 2500,
    //      int 	blockMatchingStep2 = 400,
    //      int 	groupSize = 8,
    //      int 	slidingStep = 1,
    //      float 	beta = 2.0f,
    //      int 	normType = cv::NORM_L2,
    //      int 	step = cv::xphoto::BM3D_STEPALL,
    //      int 	transformType = cv::xphoto::HAAR 
    cv::Mat Image_noisy_raw = cv::imread("Corn.pgm", cv::IMREAD_GRAYSCALE);
    cv::Mat Image_original = cv::imread("Corn_gray.pgm", cv::IMREAD_GRAYSCALE);
    cv::Mat Image_afterOp;
    // We need to crop the images
    cv::Mat Image_noisy = Image_noisy_raw(cv::Rect(2, 2, Image_width + 2, Image_height + 2));
    Image_noisy_raw.release();
    cv::xphoto::bm3dDenoising(Image_noisy,
                              Image_afterOp,
                              20, 8, 256, 2500, 400, 16, 1, 2.0f,
                              cv::NORM_L2,
                              cv::xphoto::BM3D_STEPALL,
                              cv::xphoto::HAAR);

    cv::imshow("Denoised Image", Image_afterOp);
    cv::imshow("Original", Image_original);
    cv::imshow("Noisy", Image_noisy);
    cv::waitKey();
    cv::imwrite("BM3D_Corn.pgm", Image_afterOp);
}