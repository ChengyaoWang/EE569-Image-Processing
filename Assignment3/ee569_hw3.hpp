//
//  ee569_hw3.hpp
//
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
#ifndef ee569_hw3_hpp
#define ee569_hw3_hpp

# include <stdio.h>
# include <iostream>
# include "opencv2/core.hpp"

using namespace std;



# define SHRINKING 0x00
# define THINNING 0x01
# define SKELETONIZING 0x02
# define PREPROCESSING 0x03
# define MASK_FOR_ONLY01 0xFF
# define MASK_FOR_NOABC  0x00

class Matrix_ToolBox{
    public:
        Matrix_ToolBox(){}
        ~ Matrix_ToolBox(){}
        static double Determinant_Cal(double ** scr_Mx, int n);
        static float ** Inverse_Cal(double ** scr_Mx, int n);
        static float ** Inverse_Cal(float ** scr_Mx, int n);
        static float ** Matrix_Mul(float ** left_Mx, float ** right_Mx, int * dims);
        static float  * Matrix_Mul(float * left_Vec, float ** right_Mx, int * dims);
        static float  * Matrix_Mul(float ** left_Mx, float * right_Vec, int * dims);
        static float ** Matrix_Transpose(float ** scr_Mx, int scr_M, int scr_N);
        static float ** Allocate_Mx(int M, int N);
        static float ** Allocate_Mx(int N);
        static double ** Allocate_Mx_Double(int N);
        static double ** Allocate_Mx_Double(int M, int N);
    private:
        static void Adjoint_Cal(double ** scr_Mx, double ** adj_Mx, int n);
        static void Cofactor_Cal(double ** scr_Mx, double ** temp_Mx, int p, int q, int n);

};

class ee569_hw3_sol{
    public:
        ee569_hw3_sol(){
            Transition_Mx = Matrix_ToolBox::Allocate_Mx(3);
            Transition_Mx_Rev = Matrix_ToolBox::Allocate_Mx(3);
        }
        ~ ee569_hw3_sol(){}
        bool Print_OutputMessage = false, Print_InputMessage = false, Print_TestMessage = false;
        bool Mask_UseOriginal = true;
        unsigned char Binarizing_Threshold = 128;

        void READ_ALL_RAW();
        void pxm_TO_RAW_ALL();

        unsigned char *** Geometric_Warping(string filename, int height, int width, int channel);

        int *** FeatureMatching_FLANN(string filename1, string filename2, int ControlPoint_Num, int PreSelectedIdx);
        unsigned char *** Image_Stitching(vector<string> filenameList, vector<int> heightList, vector<int> widthList,
                                          int channel, int centerImg_Idx, int ControlPoint_Num);
        void MorphologicalProcess_Basic(int lowIter, int highIter, int stepIter);
        void MorphologicalProcess_CountStars(string filename, int height, int width, int channel);
        void MorphologicalProcess_PCBanalysis(string filename, int height, int width, int channel);
        void MorphologicalProcess_DefeatDetection(string filename, int height, int width, int channel);

    private:
        float CenterImg_x, CenterImg_y;
        float ** Transition_Mx, ** Transition_Mx_Rev; // It's 3 by 3
        int A[2], B[2], C[2], D[2]; // This is the buffer to store shifting information
        unsigned char *** input_raw(string filename, int height, int width, int channels, int padding, string extra_arg, bool zero_padding);
        unsigned char *** input_pgm(string filename);
        unsigned char *** input_ppm(string filename);
        void output_pgm(string filename, unsigned char *** image, int height, int width, int channel);
        void output_ppm(string filename, unsigned char *** image, int height, int width, int channel);
        void output_raw(string filename, unsigned char *** image, int height, int width, int channel);
        unsigned char *** AllocateImg_Uchar(int height, int width, int channel);
        float *** AllocateImg_Float(int height, int width, int channel);
        bool *** AllocateImg_Bool(int height, int width, int channel);
        int *** AllocateImg_Int(int height, int width, int channel);
        // float ** kernel_init(string kernel_type, int kernel_height, int kernel_width);
        float * Cartesian_to_Index(float * cartesian);
        float * Index_to_Cartesian(int * index);
        float * Mapping_DestiCart_to_ScrIdx(float * desti_Cart, string TYPE);
        float * Mapping_ScrCart_to_DestiIdx(float * scr_Cart, string TYPE);
        unsigned char Interpolation_Bilinear(unsigned char ** scr_Img, float * scr_Coordinate, int height, int width);
        static bool Compare_Matching(cv::DMatch match1, cv::DMatch match2);
        void TransitionMx_Cal(float *** matching_Mx, int ControllPnt_Num, string TYPE);
        void TransitionMx_Cal(float theta, float i, float j);
        void Img_extend();
        unsigned char *** Image_LinearMorph_STEP(unsigned char *** scr_Img, int height, int width, int channel, bool IS_RIGHT);
        // This part is for Image Morphological Processing
        void Image_Binarize(unsigned char ** scr_Img, int height, int width);
        void Image_Reverse(unsigned char ** scr_Img, int height, int width);
        unsigned char ** Image_Crop(unsigned char ** scr_Img, int height, int width, int up , int down, int left, int right);
        unsigned char ** Image_Copy(unsigned char ** scr_Img, int height, int width);
        unsigned char *** Image_Copy(unsigned char *** scr_Img, int height, int width, int channel);

        
        vector<unsigned char> Fetch_ConditionalMask_BASE(int BOND_NUM);
        vector<unsigned char> Fetch_ConditionalMask(unsigned char TYPE);
        vector<array<unsigned char, 3>> Fetch_UnconditionalMask(bool FOR_ST);
        bool Match_ConditionalMask(vector<unsigned char> Mask_Pool, unsigned char Byte_Input);
        bool Match_UnconditionalMask(vector<array<unsigned char, 3>> Mask_Pool, unsigned char Byte_Input);
        unsigned char Reorder_Neighborhood(unsigned char ** scr_Img, int i, int j);
        void Morph_OneIter(unsigned char ** scr_Img, int height, int width, int maxIter, unsigned char TYPE);
        unsigned char *** Morph_DO(unsigned char *** scr_Img, int height, int width, int max_Iter, unsigned char TYPE,
                                   string outputName);
        int DFS(unsigned char ** scr_Img, int i, int j, int height, int width, unsigned char foodVal);
        int DFS(unsigned char ** scr_Img, int i, int j, int height, int width);

};

#endif /* ee569_hw3_hpp */