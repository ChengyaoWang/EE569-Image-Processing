//
// Created by Administrator on 2020/3/3.
//

# include "MyUtils.hpp"

# include <stdio.h>
# include <iostream>
# include <fstream>
# include <sstream>
# include <stdlib.h>
# include <stdexcept>
# include <vector>

using namespace std;

//Static class variables
int myUtils::Buff_Height;
int myUtils::Buff_Width;
int myUtils::Buff_Channel;
bool myUtils::Print_OutputMessage;
bool myUtils::Print_InputMessage;

////////////myKernel methods////////////
myKernel::myKernel(float ** Input_data, int Input_height, int Input_width){
    data_ = myUtils::AllocateImg_Float(Input_height, Input_width, 1)[0];
    for(int i = 0; i < Input_height; ++i){
        for(int j = 0; j < Input_width; ++j){
            data_[i][j] = Input_data[i][j];
        }
    }
    height = Input_height;
    width = Input_width;
}
myKernel::myKernel(int Input_height, int Input_width){
    data_ = myUtils::AllocateImg_Float(Input_height, Input_width, 1)[0];
    height = Input_height;
    width = Input_width;
}
float & myKernel::operator() (int i, int j){
    if(i < 0 || i >= height || j < 0 || j >= width)
        throw out_of_range("Matrix subcript out of bounds");
    return data_[i][j];
}
float myKernel::operator() (int i, int j) const{
    if(i < 0 || i >= height || j < 0 || j >= width)
        throw out_of_range("Matrix subcript out of bounds");
    return data_[i][j];
}
void myKernel::show(){
    cout << "The parameter of this Kernel is: " << endl;
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            cout << data_[i][j] << " ";
        }
        cout << endl;
    }
    cout << "Height: " << height << " Width: " << width << endl;
}


////////////Image Allocation////////////
unsigned char *** myUtils::AllocateImg_Uchar(int height, int width, int channel){
    unsigned char *** pnt = new unsigned char **[channel];
    for(int k = 0; k < channel; ++k){
        pnt[k] = new unsigned char *[height];
        for(int i = 0; i < height; ++i){
            pnt[k][i] = new unsigned char [width];
        }
    }
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                pnt[k][i][j] = 0;
            }
        }
    }
    return pnt;
}
int *** myUtils::AllocateImg_Int(int height, int width, int channel){
    int *** pnt = new int **[channel];
    for(int k = 0; k < channel; ++k){
        pnt[k] = new int *[height];
        for(int i = 0; i < height; ++i){
            pnt[k][i] = new int [width];
        }
    }
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                pnt[k][i][j] = 0;
            }
        }
    }
    return pnt;
}
float *** myUtils::AllocateImg_Float(int height, int width, int channel){
    float *** pnt = new float **[channel];
    for(int k = 0; k < channel; ++k){
        pnt[k] = new float *[height];
        for(int i = 0; i < height; ++i){
            pnt[k][i] = new float [width];
        }
    }
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                pnt[k][i][j] = 0.;
            }
        }
    }
    return pnt;
}
bool *** myUtils::AllocateImg_Bool(int height, int width, int channel){
    bool *** pnt = new bool **[channel];
    for(int k = 0; k < channel; ++k){
        pnt[k] = new bool *[height];
        for(int i = 0; i < height; ++i){
            pnt[k][i] = new bool [width];
        }
    }
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                pnt[k][i][j] = false;
            }
        }
    }
    return pnt;
}
////////////Input Functions////////////
unsigned char *** myUtils::input_raw(string filename, int height, int width, int channels, int padding, string extra_arg, bool zero_padding){
    unsigned char *** image_buff = AllocateImg_Uchar(height + 2 * padding, width + 2 * padding, channels);
    // Read Image
    FILE * pFile;
    pFile = fopen((filename + extra_arg +".raw").c_str(), "rb");
    if (pFile == NULL){
        cout << "Cannot open file: " << filename <<endl;
    }
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            for(int k = 0; k < channels; ++k){
                fread(image_buff[k][i + padding] + j + padding, sizeof(unsigned char), 1, pFile);
            }
        }
    }
    // Fill in the values for reflection padding
    if(!zero_padding){
        for(int k = 0; k < channels; ++k){
            for(int i = padding; i < height + padding; ++i){
                for(int p = 0; p < padding; ++p){
                    image_buff[k][i][p] = image_buff[k][i][2 * padding - p];
                    image_buff[k][i][width + 2 * padding - 1 - p] = image_buff[k][i][width - 1 + p];
                }
            }
            for(int j = 0; j < width + 2 * padding; ++j){
                for(int p = 0; p < padding; ++p){
                    image_buff[k][p][j] = image_buff[k][2 * padding - p][j];
                    image_buff[k][height + 2 * padding - 1 - p][j] = image_buff[k][height - 1 + p][j];
                }
            }
        }
    }
    if(Print_InputMessage)
        cout << "Raw Image:" << filename << extra_arg << " Read Succeedded" << endl;
    return image_buff;
}
unsigned char *** myUtils::input_pgm(string filename){
    /*
        Read pgm image, Code reference
        We are not varifying the headers for input images
        https://stackoverflow.com/questions/8126815/how-to-read-in-data-from-a-pgm-file-in-c
    */
    ifstream infile(filename + ".pgm");
    string buff;
    getline(infile,buff);
    // Get Size
    infile >> buff;  Buff_Width = stoi(buff); infile >> buff;  Buff_Height = stoi(buff);
    Buff_Channel = 1;
    getline(infile,buff); getline(infile,buff);
    unsigned char *** returnPnt = AllocateImg_Uchar(Buff_Height, Buff_Width, 1);
    for(int i = 0; i < Buff_Height; ++i){
        for(int j = 0; j < Buff_Width; ++j){
            infile >> buff;
            returnPnt[0][i][j] = (unsigned char) stoi(buff);
        }
    }
    infile.close();
    return returnPnt;
}
unsigned char *** myUtils::input_ppm(string filename){
    /*
        Read pgm image, Code reference
        We are not varifying the headers for input images
        https://stackoverflow.com/questions/8126815/how-to-read-in-data-from-a-pgm-file-in-c
    */
    ifstream infile(filename + ".ppm");
    string buff;
    getline(infile,buff);
    // Get Size
    infile >> buff;  Buff_Width = stoi(buff); infile >> buff;  Buff_Height = stoi(buff);
    Buff_Channel = 3;
    getline(infile,buff); getline(infile,buff);
    unsigned char *** returnPnt = AllocateImg_Uchar(Buff_Height, Buff_Width, 3);
    for(int i = 0; i < Buff_Height; ++i){
        for(int j = 0; j < Buff_Width; ++j){
            for(int k = 0; k < 3; ++k){
                infile >> buff;
                returnPnt[k][i][j] = (unsigned char) stoi(buff);
            }
        }
    }
    infile.close();
    return returnPnt;
}
////////////Output Functions////////////
void myUtils::output_pgm(string filename, unsigned char *** image, int height, int width, int channel){
    if(Print_OutputMessage)      cout << "Outputing PGM File for " << filename;
    ofstream pgmFile(filename + ".pgm");
    // Header
    pgmFile << "P2" << endl << width << " " << height << endl << 255 << endl;
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            pgmFile << (int) image[0][i][j] << " ";
        }
    }
    pgmFile.close();
    if(Print_OutputMessage)      cout << ".......Finished" << endl;
}
void myUtils::output_ppm(string filename, unsigned char *** image, int height, int width, int channel){
    if(Print_OutputMessage)      cout << "Outputing PPM File for " << filename;
    ofstream pgmFile(filename + ".ppm");
    pgmFile << "P3" << endl << width << " " << height << endl << 255 << endl;
    // If it's 24-bit, 3 channels RGB
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            pgmFile << (int) image[0][i][j] << " ";
            pgmFile << (int) image[1][i][j] << " ";
            pgmFile << (int) image[2][i][j] << endl;
        }
    }
    pgmFile.close();
    if(Print_OutputMessage)      cout << ".......Finished" << endl;
}
void myUtils::output_raw(string filename, unsigned char *** image, int height, int width, int channel){
    if(Print_OutputMessage)      cout << "Outputing raw File for " << filename;
    FILE * file;
    file = fopen((filename + ".raw").c_str(), "wb");
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            for(int k = 0; k < channel; ++k){
                fwrite(image[k][i] + j, sizeof(unsigned char), 1, file);
            }
        }
    }
    fclose(file);
    if(Print_OutputMessage)      cout << ".......Finished" << endl;
}
////////////Image Operations////////////
void myUtils::Image_Binarize(unsigned char ** scr_Img, int height, int width, int Threshold){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            scr_Img[i][j] = 255 * (unsigned char) (scr_Img[i][j] >= Threshold);
        }
    }
}
void myUtils::Image_Reverse(unsigned char ** scr_Img, int height, int width){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            scr_Img[i][j] = ~ scr_Img[i][j];
        }
    }
}
unsigned char ** myUtils::Image_Crop(unsigned char ** scr_Img, int height, int width, int dim[4]){
    int up = dim[0], down = dim[1], left = dim[2], right = dim[3];
    unsigned char ** ret_Img = AllocateImg_Uchar(height - up - down, width - left - right, 1)[0];
    for(int i = up; i < height - down; ++i){
        for(int j = left; j < width - right; ++j){
            ret_Img[i - up][j - left] = scr_Img[i][j];
        }
    }
    return ret_Img;
}
unsigned char ** myUtils::Image_Copy(unsigned char ** scr_Img, int height, int width){
    unsigned char ** ret_Img = AllocateImg_Uchar(height, width, 1)[0];
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            ret_Img[i][j] = scr_Img[i][j];
        }
    }
    return ret_Img;
}
void myUtils::Image_Copy(unsigned char ** scr_Img, vector<vector<float>> & tar_Img, int upLeft_i, int upLeft_j){
    for(int i = 0; i < tar_Img.size(); ++i){
        for(int j = 0; j < tar_Img[0].size(); ++j){
            tar_Img[i][j] = (float) scr_Img[upLeft_i + i][upLeft_j + j];
        }
    }
}
unsigned char *** myUtils::Image_Copy(unsigned char *** scr_Img, int height, int width, int channel){
    unsigned char *** ret_Img = AllocateImg_Uchar(height, width, channel);
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                ret_Img[k][i][j] = scr_Img[k][i][j];
            }
        }
    }
    return ret_Img;
}
float *** myUtils::Image_Pad(float *** scr_Img, int height, int width, int channel, int dim[4], bool zeroPadding){
    int up = dim[0], down = dim[1], left = dim[2], right = dim[3];
    /*
        We Always do Reflection Extention
    */
    float *** ret_Img = AllocateImg_Float(height + up + down, width + left + right, channel);
    // Copy Image
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                ret_Img[k][i + up][j + left] = scr_Img[k][i][j];
            }
        }
    }
    // Do Padding
    if(!zeroPadding){
        for(int k = 0; k < channel; ++k){
            for(int i = up; i < height + up; ++i){
                for(int p = 0; p < left; ++p){
                    ret_Img[k][i][p] = ret_Img[k][i][2 * left - p];
                }
                for(int p = 0; p < right; ++p){
                    ret_Img[k][i][width + left + right - 1 - p] = ret_Img[k][i][width + left - right - 1 + p];
                }
            }
            for(int j = 0; j < width + left + right; ++j){
                for(int p = 0; p < up; ++p){
                    ret_Img[k][p][j] = ret_Img[k][2 * up - p][j];
                }
                for(int p = 0; p < down; ++p){
                    ret_Img[k][height + up + down - 1 - p][j] = ret_Img[k][height + up - down - 1 + p][j];
                }
            }
        }
    }
    return ret_Img;
}
float *** myUtils::Image_Pad(unsigned char *** scr_Img, int height, int width, int channel, int dim[4], bool zeroPadding){
    int up = dim[0], down = dim[1], left = dim[2], right = dim[3];
    /*
        We Always do Reflection Extention
    */
    float *** ret_Img = AllocateImg_Float(height + up + down, width + left + right, channel);
    // Copy Image
    for(int k = 0; k < channel; ++k){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                ret_Img[k][i + up][j + left] = (float) scr_Img[k][i][j];
            }
        }
    }
    // Do Padding
    if(!zeroPadding){
        for(int k = 0; k < channel; ++k){
            for(int i = up; i < height + up; ++i){
                for(int p = 0; p < left; ++p){
                    ret_Img[k][i][p] = ret_Img[k][i][2 * left - p];
                }
                for(int p = 0; p < right; ++p){
                    ret_Img[k][i][width + left + right - 1 - p] = ret_Img[k][i][width + left - right - 1 + p];
                }
            }
            for(int j = 0; j < width + left + right; ++j){
                for(int p = 0; p < up; ++p){
                    ret_Img[k][p][j] = ret_Img[k][2 * up - p][j];
                }
                for(int p = 0; p < down; ++p){
                    ret_Img[k][height + up + down - 1 - p][j] = ret_Img[k][height + up - down - 1 + p][j];
                }
            }
        }
    }
    return ret_Img;
}
unsigned char ** myUtils::Image_Pad(unsigned char ** scr_Img, int height, int width, int dim[4], bool zeroPadding){
    int up = dim[0], down = dim[1], left = dim[2], right = dim[3];
    /*
        We Always do Reflection Extention
    */
    unsigned char ** New_Img = AllocateImg_Uchar(height + up + down, width + left + right, 1)[0];
    // Copy Image
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            New_Img[i + up][j + left] = scr_Img[i][j];
        }
    }
    // Do Padding
    if(!zeroPadding){
        for(int i = up; i < height + up; ++i){
            for(int p = 0; p < left; ++p){
                New_Img[i][p] = New_Img[i][2 * left - p];
            }
            for(int p = 0; p < right; ++p){
                New_Img[i][width + left + right - 1 - p] = New_Img[i][width + left - right - 1 + p];
            }
        }
        for(int j = 0; j < width + left + right; ++j){
            for(int p = 0; p < up; ++p){
                New_Img[p][j] = New_Img[2 * up - p][j];
            }
            for(int p = 0; p < down; ++p){
                New_Img[height + up + down - 1 - p][j] = New_Img[height + up - down - 1 + p][j];
            }
        }
    }
    return New_Img;
}
//////////////Type Cast/////////////////



////////////Script////////////////////////
/*
    void myUtils::READ_ALL_RAW(vector<string> filenameList, vector<array<int, 3>> dimList, string scrPath, string destiPath){
        if(filenameList.size() != dimList.size())
            throw "The size doesn't match";
        unsigned char *** buff_Img;
        for(int i = 0; i < filenameList.size(); ++i){
            buff_Img = input_raw(scrPath + filenameList[i], dimList[i][0], dimList[i][1], dimList[i][2], 0, "", true);
            if(dimList[i][2] == 1)
                output_pgm(destiPath + filenameList[i], buff_Img, dimList[i][0], dimList[i][1], dimList[i][2]);
            else if(dimList[i][2] == 3)
                output_ppm(destiPath + filenameList[i], buff_Img, dimList[i][0], dimList[i][1], dimList[i][2]);
        }

    }
    void myUtils::PXM_TO_RAW_ALL(vector<string> filenameList, vector<int> channelList, string scrPath, string destiPath){
        if(filenameList.size() != channelList.size())
            throw "The size doesn't match";
        unsigned char *** buff_Img;
        for(int i = 0; i < filenameList.size(); ++i){
            if(channelList[i] == 1)
                buff_Img = input_pgm(scrPath + filenameList[i]);
            else if(channelList[i] == 3)
                buff_Img = input_ppm(scrPath + filenameList[i]);
            output_raw(destiPath + filenameList[i], buff_Img, Buff_Height, Buff_Width, Buff_Channel);
        }
    }
*/