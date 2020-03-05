//
//  ee569_hw1.hpp
//  opencv_playground
//
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
#ifndef ee569_hw2_hpp
#define ee569_hw2_hpp

# include <stdio.h>
# include <iostream>

// Define constants for MBVQ Color diffusion
// Use 8 bits to encode Quadruples information
// White(7) / Yellow(6) / Cyan(5) / Magenta(4) / Green(3) / Red(2) / Blue(1) / Black(0)
# define CMYW 0xF0; // 11110000 
# define MYGC 0x78; // 01111000
# define RGMY 0x5C; // 01011100; 
# define KRGB 0x0F; // 00001111; 
# define RGBM 0x1E; // 00011110; 
# define CMGB 0x3A; // 00111010; 


using namespace std;

class ee569_hw2_sol{
    /*
        1.
            For Assignment 2, We've decided not to define the main Image of the function class
            Every Image only stays within the scope of relevant functions
        2.
            Gray scale images in principle should be a 2D matrix, but here we implement as 3D
            From Homework 3, dimensions of Image will be organized as Depth * Height * Width for numerous reasons
        3.
            Paremter Tuning process is embeded in the Post-Op image generating process, shows f1 score
    */
    public:
        ee569_hw2_sol(){
            cout << "Reading Ground Truth Edges for Gallery / Dogs" << endl;
            string path = "./GroundTruth/";
            Edge_truth_pool = new unsigned char ****[2];
            Edge_truth_pool[0] = new unsigned char ***[5];
            Edge_truth_pool[1] = new unsigned char ***[5];
            for(int i = 0; i < 5; i ++){
                Edge_truth_pool[0][i] = input_pgm(path + "Gallery_GT" + to_string(i + 1));
                Edge_truth_pool[1][i] = input_pgm(path + "Dogs_GT" + to_string(i + 1));
            }
        }
        ~ ee569_hw2_sol(){
            delete Edge_truth_pool;
        }
        bool print_f1 = false, print_pgmMessage = true, print_pgmppmImage = true;
        void Sobel_edge_detector(string filename, int height, int width, int channel, bool toy_normal);
        void Canny_edge_detector(string filename, int height, int width, int channel);
        void Structured_edge_detector(string filename, int height, int width, int channel, bool img_show);
        void best_setting_show(string filename, int height, int width, int channel);
        void Dithering_fixed_T(string filename, int height, int width, int channel);
        void Dithering_random_T(string filename, int height, int width, int channel);
        void Dithering_Mx(string filename, int height, int width, int channel, int kernel_dim);
        void Error_diffusion_Floyd(string filename, int height, int width, int channel);
        void Error_diffusion_JJN(string filename, int height, int width, int channel);
        void Error_diffusion_Stucki(string filename, int height, int width, int channel);
        void Error_diffusion_NaiveColored(string filename, string TYPE, int height, int width, int channel);
        void Error_diffusion_MBVQ(string filename, string TYPE, int height, int width, int channel);
        void EDGE_DETECTOR_VISUALIZE_ALL();
        void EDGE_DETECTOR_DO_ALL(bool f1, bool pgmMessage, bool pgmppmImage);
        void HALF_TUNE_DO_ALL();
        void pxm_to_raw_DO_ALL();
        void f1_test();

    private:
        unsigned char ***** Edge_truth_pool;
        double MixUp_R = 0.2989, MixUp_G = 0.5870, MixUp_B = 0.1140;
        unsigned char mapping[8][3] = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0},
                                       {1, 0, 1}, {0, 1, 1}, {1, 1, 0}, {1, 1, 1}};
        unsigned char *** input_raw(string filename, int height, int width, int channels, int padding, string extra_arg, bool zero_padding);
        unsigned char *** input_pgm(string filename);
        unsigned char *** input_ppm(string filename);
        void output_pgm(string filename, unsigned char *** image, int height, int width, int channel);
        void output_ppm(string filename, unsigned char *** image, int height, int width, int channel);
        void output_raw(string filename, unsigned char *** image, int height, int width, int channel);
        unsigned char *** allocate_Image(int height, int width, int channel);
        float *** allocate_Image_float(int height, int width, int channel);
        bool *** allocate_Image_bool(int height, int width, int channel);
        float fixed_kernel_CONV(float ** kernel, int kernel_size, unsigned char *** image, int upperleft_i, int upperleft_j, int k);
        void Error_diffuse_step(float *** Image, float ** kernel, int kernel_size,
                                int upperleft_i, int upperleft_j, int channel, float * error);
        unsigned char *** Error_diffuse_DO(float *** kernel, int padding, float *** scr_Image, float T,
                                             int height, int width, int channel, string TYPE);
        float ** kernel_init(string kernel_type, int kernel_height, int kernel_width);
        unsigned char *** rbg_to_gray(unsigned char *** image_rgb, int height, int width);
        double f1_score_cal(unsigned char *** pred_image, string filename, int height, int width, int TYPE);
        unsigned char fetch_quadruples(float * location);
        unsigned char * fetch_closest(string TYPE, float * RGB_at_ij, int channel, float T);
        void edge_error_visualization(string filename, string Target, int height, int width, int channel);
        bool *** correspondPixel(unsigned char *** A, unsigned char *** B, int height, int width, int neighborSize);
};
#endif /* ee569_hw1_hpp */
