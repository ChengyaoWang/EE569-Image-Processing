//
//  ee569_hw1.hpp
//  opencv_playground
//
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
#ifndef ee569_hw1_hpp
#define ee569_hw1_hpp

# include <stdio.h>
# include <iostream>


using namespace std;

class ee569_hw1_sol{
    public:
        ee569_hw1_sol(){
//            cout << "//////////////////////////////////////////////////////////////////////" << endl;
//            cout << "EE569_hw1_sol Class instanitated" << endl;
//            cout << "Futher Notice: utils function will be moved to Parent Class Begining from hw2" << endl;
//            cout << "Only Functions that are direct answer to homework problems will stay" << endl;
//            cout << "Assuptions:" << endl << "The Upper-left most pixel for raw images will be taken as Green" << endl;
//            cout << "//////////////////////////////////////////////////////////////////////" << endl;
        }
        ~ ee569_hw1_sol(){
            delete Image;
        }
        void read_image_raw(string filename, int width, int height, int channels);
        void print_ppm(string target_filename, unsigned char *** target_image, int target_height, int target_width, int target_channel);
        void print_pgm(string target_filename, unsigned char *** target_image, int target_height, int target_width, int target_channel);
        void output_raw(string filename, unsigned char *** image, int channel);
        void parameter_tuning(bool stage_select[]);
        void demosaicing_bilinear();
        void demosaicing_MHC();
        void enhancement_transfer_function();
        void enhancement_cumulative_p();
        double denoising_uniform(int kernel_size, bool verbose);
        double denoising_gaussian(int kernel_size, float std_a, bool verbose);
        double denoising_bilateral(int kernel_size, float std_a, float std_s, bool verbose);
        double denoising_NLM(float std_a, float std_h, bool verbose, int corpus_size);
        void denoising_BM3D();
        

    private:
        unsigned char *** Image;
        unsigned char *** Image_noiseless;
        int Image_width, Image_height, Image_channels;
        string Image_filename;
        float MHC_alpha = 1./ 2, MHC_beta = 5./ 8, MHC_gamma = 3./ 4;
        unsigned char conv_op_demoisaic(int center_i, int center_j, string kernel_type, string target_color);
        unsigned char conv_op_denoise_fixedKernel(int center_i, int center_j, int channel, float ** kernel);
        unsigned char conv_op_denoise_bilateral(int center_i, int center_j, int channel, float std_s, float ** kernel);
        unsigned char conv_op_denoise_NLM(int center_i, int center_j, int channel, int window_size, float h, float a);
        unsigned char *** allocate_Image(int height, int width, int channel);
        unsigned char *** image_reflect_extend(int size_added);
        float ** kernel_init(string kernel_name, int size, string target_color, string center_color, double std_c);
        double metric_cal(double x_diff, double y_diff, string metric_type);
        double PSNR_cal(unsigned char *** raw_image, unsigned char *** denoised_image);
};
#endif /* ee569_hw1_hpp */
