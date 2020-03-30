//
// Created by Administrator on 2020/3/3.
//

#ifndef MYUTILS_HPP
#define MYUTILS_HPP

# include <stdio.h>
# include <iostream>

class myKernel{
    public:
        myKernel(float ** Input_data, int Input_height, int Input_width);
        myKernel(int Input_height, int Input_width);
        ~myKernel(){}
        float& operator() (int i, int j);
        float  operator() (int i, int j) const;
        void show();
        float ** data_;
        int height, width;
    protected:
    private:
};

class myUtils{
    public:
        myUtils(){}
        ~myUtils(){}

        static int Buff_Height, Buff_Width, Buff_Channel;
        static bool Print_OutputMessage, Print_InputMessage;

        static unsigned char *** AllocateImg_Uchar(int height, int width, int channel);
        static int           *** AllocateImg_Int(int height, int width, int channel);
        static float         *** AllocateImg_Float(int height, int width, int channel);
        static bool          *** AllocateImg_Bool(int height, int width, int channel);

        static unsigned char *** input_raw(std::string filename, int height, int width, int channels,
                                           int padding, std::string extra_arg, bool zero_padding);
        static unsigned char *** input_pgm(std::string filename);
        static unsigned char *** input_ppm(std::string filename);

        static void output_pgm(std::string filename, unsigned char *** image, int height, int width, int channel);
        static void output_ppm(std::string filename, unsigned char *** image, int height, int width, int channel);
        static void output_raw(std::string filename, unsigned char *** image, int height, int width, int channel);

        static void Image_Binarize(unsigned char ** scr_Img, int height, int width, int Threshold);
        static void Image_Reverse(unsigned char ** scr_Img, int height, int width);
        static unsigned char ** Image_Crop(unsigned char ** scr_Img, int height, int width, int dim[4]);
        
        static float *** Image_Pad(float *** scr_Img, int height, int width, int channel, int dim[4], bool zeroPadding);
        static float *** Image_Pad(unsigned char *** scr_Img, int height, int width, int channel, int dim[4], bool zeroPadding);
        static unsigned char ** Image_Pad(unsigned char ** scr_Img, int height, int width, int dim[4], bool zeroPadding);

        static unsigned char **  Image_Copy(unsigned char ** scr_Img, int height, int width);
        static unsigned char *** Image_Copy(unsigned char *** scr_Img, int height, int width, int channel);
        static void Image_Copy(unsigned char ** scr_Img, std::vector<std::vector<float>> & tar_Img, int upLeft_i, int upLeft_j);

        // static void READ_ALL_RAW(vector<string> filenameList, vector<array<int, 3>> dimList, string scrPath, string destiPath);
        // static void PXM_TO_RAW_ALL(vector<string> filenameList, vector<int> channelList,  string scrPath, string destiPath);


    protected:
    private:
};

#endif
