//  ee569_hw3.cpp
//  Created by Chengyao Wang on 1/16/20.
//  Copyright © 2020 Chengyao Wang. All rights reserved.
//
# include "ee569_hw3.hpp"
# include <stdio.h>
# include <iostream>
# include <fstream>
# include <sstream>
# include <stdlib.h>
# include <math.h>

# include "opencv2/core.hpp"
# include "opencv2/highgui.hpp"
# include "opencv2/features2d.hpp"
# include "opencv2/xfeatures2d.hpp"


using namespace std;

float ** Matrix_ToolBox::Allocate_Mx(int M, int N){
    float ** ret_Mx = new float * [M];
    for(int i = 0; i < M; ++i){
        ret_Mx[i] = new float [N];
    }
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            ret_Mx[i][j] = 0.;
        }
    }
    return ret_Mx;
}
float ** Matrix_ToolBox::Allocate_Mx(int N){
    float ** ret_Mx = new float * [N];
    for(int i = 0; i < N; ++i){
        ret_Mx[i] = new float [N];
    }
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            ret_Mx[i][j] = 0.;
        }
    }
    return ret_Mx;
}
double ** Matrix_ToolBox::Allocate_Mx_Double(int M, int N){
    double ** ret_Mx = new double * [M];
    for(int i = 0; i < M; ++i){
        ret_Mx[i] = new double [N];
    }
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            ret_Mx[i][j] = 0.;
        }
    }
    return ret_Mx;
}
double ** Matrix_ToolBox::Allocate_Mx_Double(int N){
    double ** ret_Mx = new double * [N];
    for(int i = 0; i < N; ++i){
        ret_Mx[i] = new double [N];
    }
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            ret_Mx[i][j] = 0.;
        }
    }
    return ret_Mx;
}
void Matrix_ToolBox::Cofactor_Cal(double ** scr_Mx, double ** temp_Mx, int p, int q, int n){
    int i = 0, j = 0; 
    for (int row = 0; row < n; row++){ 
        for (int col = 0; col < n; col++){ 
            if (row != p && col != q) { 
                temp_Mx[i][j++] = scr_Mx[row][col]; 
                if (j == n - 1){ 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 
}
double Matrix_ToolBox::Determinant_Cal(double ** scr_Mx, int n){
    double D = 0; // Initialize result 
    if (n == 1) 
        return scr_Mx[0][0]; 
    double ** temp_Mx = Allocate_Mx_Double(n);
    int sign = 1;  // To store sign multiplier 
    for (int f = 0; f < n; f++){ 
        Cofactor_Cal(scr_Mx, temp_Mx, 0, f, n); 
        D += sign * scr_Mx[0][f] * Determinant_Cal(temp_Mx, n - 1); 
        sign = - sign; 
    } 
    return D;
} 
void Matrix_ToolBox::Adjoint_Cal(double ** scr_Mx, double ** adj_Mx, int n){
    if (n == 1){ 
        adj_Mx[0][0] = 1.;
        return;
    }
    int sign = 1;
    double ** temp_Mx = Allocate_Mx_Double(n);
    for (int i=0; i < n; i++){ 
        for (int j=0; j < n; j++){
            Cofactor_Cal(scr_Mx, temp_Mx, i, j, n); 
            sign = ((i+j)%2==0)? 1: -1; 
            adj_Mx[j][i] = (sign) * (Determinant_Cal(temp_Mx, n - 1)); 
        } 
    } 
}
float ** Matrix_ToolBox::Inverse_Cal(double ** scr_Mx, int n){
    double det = Determinant_Cal(scr_Mx, n); 
    double ** inv_Mx = Allocate_Mx_Double(n), ** adj_Mx = Allocate_Mx_Double(n);
    float ** ret_Mx = Allocate_Mx(n);
    if (det == 0) { 
        cout << "Singular matrix, can't find its inverse"; 
        return ret_Mx;
    } 
    Adjoint_Cal(scr_Mx, adj_Mx, n); 
    for(int i=0; i < n; ++i){
        for(int j=0; j < n; ++j) 
            inv_Mx[i][j] = adj_Mx[i][j] / det;
    }
    // Cast to Float 
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            ret_Mx[i][j] = (float) inv_Mx[i][j];
        }
    }
    return ret_Mx;
}
float ** Matrix_ToolBox::Inverse_Cal(float ** scr_Mx, int n){
    double ** Input_Mx = Allocate_Mx_Double(n);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            Input_Mx[i][j] = (double) scr_Mx[i][j];
        }
    }
    return Inverse_Cal(Input_Mx, n);
}
// Three overloaded functions to accomodate vector * matrix multiplication
float ** Matrix_ToolBox::Matrix_Mul(float ** left_Mx, float ** right_Mx, int * dims){
    int left_M = dims[0], left_N = dims[1], right_M = dims[2], right_N = dims[3];
    if(left_N != right_M)
        throw "The Dimensions Doesn't Match";
    float ** ret_Mx = Allocate_Mx(left_M, right_N), buff;
    for(int i = 0 ; i < left_M; ++i){
        for(int j = 0; j < right_N; ++j){
            buff = 0.;
            for(int k = 0; k < left_N; ++k)
                buff += left_Mx[i][k] * right_Mx[k][j];
            ret_Mx[i][j] = buff;
        }
    }
    return ret_Mx;
}
float * Matrix_ToolBox::Matrix_Mul(float * left_Vec, float ** right_Mx, int * dims){
    int left_M = dims[0], left_N = dims[1], right_M = dims[2], right_N = dims[3];
    if(left_N != right_M)
        throw "The Dimensions Doesn't Match";
    float * ret_Vec = new float [right_N], buff;
    for(int j = 0; j < right_N; ++j){
        buff = 0.;
        for(int k = 0; k < left_N; ++k)
            buff += left_Vec[k] * right_Mx[k][j];
        ret_Vec[j] = buff;
    }
    return ret_Vec;
}
float * Matrix_ToolBox::Matrix_Mul(float ** left_Mx, float * right_Vec, int * dims){
    int left_M = dims[0], left_N = dims[1], right_M = dims[2], right_N = dims[3];
    if(left_N != right_M)
        throw "The Dimensions Doesn't Match";
    float * ret_Vec = new float [left_M], buff;
    for(int i = 0 ; i < left_M; ++i){
            buff = 0.;
            for(int k = 0; k < left_N; ++k)
                buff += left_Mx[i][k] * right_Vec[k];
            ret_Vec[i] = buff;
    }
    return ret_Vec;
}
float ** Matrix_ToolBox::Matrix_Transpose(float ** scr_Mx, int scr_M, int scr_N){
    float ** ret_Mx = Allocate_Mx(scr_N, scr_M);
    for(int i = 0; i < scr_M; ++i){
        for(int j = 0; j < scr_N; ++j){
            ret_Mx[j][i] = scr_Mx[i][j];
        }
    }
    return ret_Mx;
}

unsigned char *** ee569_hw3_sol::AllocateImg_Uchar(int height, int width, int channel){
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
int *** ee569_hw3_sol::AllocateImg_Int(int height, int width, int channel){
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
float *** ee569_hw3_sol::AllocateImg_Float(int height, int width, int channel){
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
bool *** ee569_hw3_sol::AllocateImg_Bool(int height, int width, int channel){
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
unsigned char *** ee569_hw3_sol::input_raw(string filename, int height, int width, int channels, int padding, string extra_arg, bool zero_padding){
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
unsigned char *** ee569_hw3_sol::input_pgm(string filename){
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
    unsigned char *** returnPnt = AllocateImg_Uchar(height, width, 1);
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            infile >> buff;
            returnPnt[0][i][j] = (unsigned char) stoi(buff);
        }
    }
    infile.close();
    return returnPnt;
}
unsigned char *** ee569_hw3_sol::input_ppm(string filename){
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
    unsigned char *** returnPnt = AllocateImg_Uchar(height, width, 3);
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            for(int k = 0; k < 3; ++k){
                infile >> buff;
                returnPnt[k][i][j] = (unsigned char) stoi(buff);
            }
        }
    }
    infile.close();
    return returnPnt;
}
void ee569_hw3_sol::output_pgm(string filename, unsigned char *** image, int height, int width, int channel){
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
void ee569_hw3_sol::output_ppm(string filename, unsigned char *** image, int height, int width, int channel){
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
void ee569_hw3_sol::output_raw(string filename, unsigned char *** image, int height, int width, int channel){
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////This is the section for Homography transformation//////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float * ee569_hw3_sol::Cartesian_to_Index(float * cartesian){
    float * ret_idx = new float [3];
    ret_idx[0] = cartesian[0] - 1./2;
    ret_idx[1] = cartesian[1] - 1./2;
    ret_idx[2] = cartesian[2];
    return ret_idx;
}

float * ee569_hw3_sol::Index_to_Cartesian(int * index){
    float * ret_Cart = new float [3];
    ret_Cart[0] = (float) index[0] + 1./2;
    ret_Cart[1] = (float) index[1] + 1./2;
    ret_Cart[2] = (float) index[2];
    return ret_Cart;
}

float * ee569_hw3_sol::Mapping_DestiCart_to_ScrIdx(float * desti_Cart, string TYPE){
    float x = desti_Cart[0], y = desti_Cart[1];
    float * scr_Coordinate;
    if(TYPE == "Q1"){
        scr_Coordinate = new float [3];
        scr_Coordinate[0] = x;
        scr_Coordinate[1] = CenterImg_y - (CenterImg_y - y) / sqrt(1 - pow(1 - x / CenterImg_x, 2));
    }
    else if(TYPE == "Q2"){
        int * dims = new int [4] {3, 3, 3, 1};
        scr_Coordinate = Matrix_ToolBox::Matrix_Mul(Transition_Mx_Rev, desti_Cart, dims);
        scr_Coordinate[0] /= scr_Coordinate[2]; scr_Coordinate[1] /= scr_Coordinate[2];
        
    }
    return Cartesian_to_Index(scr_Coordinate);
}

float * ee569_hw3_sol::Mapping_ScrCart_to_DestiIdx(float * scr_Cart, string TYPE){
    float x = scr_Cart[0], y = scr_Cart[1];
    float * desti_Coordinate;
    if(TYPE == "Q1"){
        desti_Coordinate = new float [3];
        desti_Coordinate[0] = x;
        desti_Coordinate[1] = CenterImg_y - (CenterImg_y - y) * sqrt(1 - pow(1 - x / CenterImg_x, 2));
    }
    else if (TYPE == "Q2"){
        int * dims = new int [4] {3, 3, 3, 1};
        desti_Coordinate = Matrix_ToolBox::Matrix_Mul(Transition_Mx, scr_Cart, dims);
        desti_Coordinate[0] /= desti_Coordinate[2]; desti_Coordinate[1] /= desti_Coordinate[2];
    }
    return Cartesian_to_Index(desti_Coordinate);
}

unsigned char ee569_hw3_sol::Interpolation_Bilinear(unsigned char ** scr_Img, float * scr_Coordinate, int height, int width){
    int UpperLeft_X = floor(scr_Coordinate[0]), UpperLeft_Y = floor(scr_Coordinate[1]);
    float ret_Val = 0;
    if(UpperLeft_X >= 0 && UpperLeft_X <= height - 2 && UpperLeft_Y >= 0 && UpperLeft_Y <= width - 2){
        float shift_x = scr_Coordinate[0] - UpperLeft_X;
        float shift_y = scr_Coordinate[1] - UpperLeft_Y;
        ret_Val +=  (1 - shift_x) * (1 - shift_y) * scr_Img[UpperLeft_X][UpperLeft_Y] + 
                    shift_x * (1 - shift_y) * scr_Img[UpperLeft_X + 1][UpperLeft_Y] + 
                    (1 - shift_x) * shift_y * scr_Img[UpperLeft_X][UpperLeft_Y + 1] + 
                    shift_x * shift_y * scr_Img[UpperLeft_X + 1][UpperLeft_Y + 1];
    }
    // Mannual Cropping is actually not needed
    if(ret_Val >= 255.)     ret_Val = 255;
    else if(ret_Val < 0.)   ret_Val = 0;
    return (unsigned char)ret_Val;
}

unsigned char *** ee569_hw3_sol::Geometric_Warping(string filename, int height, int width, int channel){
    unsigned char *** scr_Img = input_raw(filename, height, width, channel, 0, "", true);
    unsigned char *** warped_Img = AllocateImg_Uchar(height, width, channel);
    CenterImg_x = height / 2; CenterImg_y = width / 2;

    float * Buff_Coordinate;
    int * Buff_Idx = new int [2];
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            Buff_Idx[0] = i; Buff_Idx[1] = j;
            Buff_Coordinate = Mapping_DestiCart_to_ScrIdx(Index_to_Cartesian(Buff_Idx), "Q1");
            for(int k = 0; k < channel; ++k){
                warped_Img[k][i][j] = Interpolation_Bilinear(scr_Img[k], Buff_Coordinate, height, width);
            }
        }
    }
    output_ppm(filename + "_Warped", warped_Img, height, width, channel);
    // Reverse the warped image back
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            Buff_Idx[0] = i; Buff_Idx[1] = j;
            Buff_Coordinate = Mapping_ScrCart_to_DestiIdx(Index_to_Cartesian(Buff_Idx), "Q1");
            for(int k = 0; k < channel; ++k){
                scr_Img[k][i][j] = Interpolation_Bilinear(warped_Img[k], Buff_Coordinate, height, width);
            }
        }
    }
    output_ppm(filename + "_RevWarped", scr_Img, height, width, channel);
    return warped_Img;
}

inline bool ee569_hw3_sol::Compare_Matching(cv::DMatch match1, cv::DMatch match2){
    return bool(match1.distance < match2.distance);
}

int *** ee569_hw3_sol::FeatureMatching_FLANN(string filename_1, string filename_2, int ControlPoint_Num, int PreSelectedIdx){
    /*
        Reference Code OpenCV Official Library
        https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
    */
    cv::Mat img1 = imread(filename_1 + "_ori.ppm", cv::IMREAD_COLOR);
    cv::Mat img2 = imread(filename_2 + "_ori.ppm", cv::IMREAD_COLOR);
    // This counts the number of SURF + FLANN done
    static int MATCHING_DONE = 0;
    if( img1.empty() || img2.empty() )
        throw "Could not open or find the image!\n";
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector -> detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
    detector -> detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    vector< vector<cv::DMatch> > knn_matches;
    matcher -> knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++){
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    sort(good_matches.begin(), good_matches.end(), Compare_Matching);
    //-- Draw matches
    vector<cv::DMatch> temp_Matches;
    cv::Mat img_matches;
    for(int i = 0; i < good_matches.size(); ++i){
        temp_Matches.push_back(good_matches[i]);
        drawMatches(img1, keypoints1, img2, keypoints2, temp_Matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::imwrite("../Stitching/matches" + to_string(MATCHING_DONE)+ "|" + to_string(i) + ".ppm", img_matches);
        //-- Show detected matches
        if(Print_OutputMessage){
            cv::imshow("Good Matches", img_matches);
            cv::waitKey();
        }
        temp_Matches.pop_back();
    }
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imwrite("../Stitching/matches" + to_string(MATCHING_DONE) + "|all.ppm", img_matches);
        //-- Show detected matches
    if(Print_OutputMessage){
        cv::imshow("Good Matches", img_matches);
        cv::waitKey();
    }
    // Constructing Return Matrix
    // keypoints1 <=> DMatch.queryIdx;
    // keypoints2 <=> DMatch.trainIdx;
    int *** ret_Matching = AllocateImg_Int(ControlPoint_Num, 3, 2), Idx;
    vector<int> points;
    if(PreSelectedIdx == 0){
        points = {16, 18, 62, 105};
    }
    else if(PreSelectedIdx == 1){
        points = {19, 46, 59, 148};
    }
    for(int i = 0; i < 4; ++i){
        Idx = points[i];
        ret_Matching[0][i][0] = keypoints1[good_matches[Idx].queryIdx].pt.y;
        ret_Matching[0][i][1] = keypoints1[good_matches[Idx].queryIdx].pt.x;
        ret_Matching[0][i][2] = 1;
        ret_Matching[1][i][0] = keypoints2[good_matches[Idx].trainIdx].pt.y;
        ret_Matching[1][i][1] = keypoints2[good_matches[Idx].trainIdx].pt.x;
        ret_Matching[1][i][2] = 1;
        temp_Matches.push_back(good_matches[Idx]);
    }
    // Print out selected features
    drawMatches( img1, keypoints1, img2, keypoints2, temp_Matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imwrite("../Stitching/matches"+ to_string(MATCHING_DONE) + "Selected.ppm", img_matches);
    // // This part serves for quick validation in Python
    // for(int i = 0; i < ControlPoint_Num; ++i){
    //     cout << "[" << ret_Matching[0][i][0] << "," << ret_Matching[0][i][1] << ",";
    //     cout << ret_Matching[0][i][2] << "]," << endl;
    // }
    // for(int i = 0; i < ControlPoint_Num; ++i){
    //     cout << "[" << ret_Matching[1][i][0] << "," << ret_Matching[1][i][1] << ",";
    //     cout << ret_Matching[1][i][2] << "]," << endl;
    // }
    MATCHING_DONE += 1;
    return ret_Matching;
}

void ee569_hw3_sol::TransitionMx_Cal(float *** matching_Mx, int ControllPnt_Num, string TYPE){
    if(ControllPnt_Num != 4)
        throw "Currently we only accept 4 control points";
    // Contruct the transition Mx
    float ** Xin = matching_Mx[0], ** Xout = matching_Mx[1];
    double ** dataMx = Matrix_ToolBox::Allocate_Mx_Double(2 * ControllPnt_Num, 8);
    float * yMx = new float [2 * ControllPnt_Num];
    for(int i = 0; i < ControllPnt_Num; ++i){
        // Odd row
        dataMx[2 * i][0] = Xin[i][0]; dataMx[2 * i][1] = Xin[i][1]; dataMx[2 * i][2] = Xin[i][2];
        dataMx[2 * i][6] = - Xout[i][0] * Xin[i][0]; dataMx[2 * i][7] = - Xout[i][0] * Xin[i][1];
        yMx[2 * i] = Xout[i][0];
        // Even row
        dataMx[2 * i + 1][3] = Xin[i][0]; dataMx[2 * i + 1][4] = Xin[i][1]; dataMx[2 * i + 1][5] = Xin[i][2];
        dataMx[2 * i + 1][6] = - Xout[i][1] * Xin[i][0]; dataMx[2 * i + 1][7] = - Xout[i][1] * Xin[i][1];
        yMx[2 * i + 1] = Xout[i][1];
    }
    int * dims = new int [4] {8, 2 * ControllPnt_Num, 2 * ControllPnt_Num, 1};
    yMx = Matrix_ToolBox::Matrix_Mul(Matrix_ToolBox::Inverse_Cal(dataMx, 8), yMx, dims);
    // delete[] dataMx;
    // Assign Values to the Transition Matrix
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            if(i == 2 && j == 2)
                break;
            Transition_Mx[i][j] = yMx[3 * i + j];
        }
    }
    Transition_Mx[2][2] = 1.;
    Transition_Mx_Rev = Matrix_ToolBox::Inverse_Cal(Transition_Mx, 3);
}

unsigned char *** ee569_hw3_sol::Image_LinearMorph_STEP(unsigned char *** scr_Img,
                                                        int height, int width, int channel, bool IS_RIGHT){
    // Get the four corners, A = Upperleft_X, B = Upperleft_Y, C = LowerRight_X, D = LowerRight_Y
    float * Buff_Coordinate = new float [3] {0., 0., 1.};
    int * dims = new int [4] {1, 3, 3, 3};
    float ** Corners = Matrix_ToolBox::Allocate_Mx(4, 3);
    Corners[0][0] = 0.5; Corners[0][1] = 0.5; Corners[0][2] = 1; 
    Corners[1][0] = 0.5; Corners[1][1] = width - 0.5; Corners[1][2] = 1; 
    Corners[2][0] = height - 0.5; Corners[2][1] = 0.5; Corners[2][2] = 1;
    Corners[3][0] = height - 0.5; Corners[3][1] = width - 0.5; Corners[3][2] = 1;
    cout << "This is the new location of the four corners " << endl;
    for(int i = 0; i < 4; ++i){
        Buff_Coordinate = Mapping_ScrCart_to_DestiIdx(Corners[i], "Q2");
        cout << Corners[i][0] << " " << Corners[i][1] << " " << Corners[i][2] << endl;
        cout << Buff_Coordinate[0] << " " << Buff_Coordinate[1] << " " << Buff_Coordinate[2] << endl;
        if(i == 0){
            A[IS_RIGHT] = floor(Buff_Coordinate[0]); B[IS_RIGHT] = floor(Buff_Coordinate[1]);
            C[IS_RIGHT] = ceil(Buff_Coordinate[0]); D[IS_RIGHT] = ceil(Buff_Coordinate[1]);
        }
        A[IS_RIGHT] = min(A[IS_RIGHT], (int) floor(Buff_Coordinate[0]));
        B[IS_RIGHT] = min(B[IS_RIGHT], (int) floor(Buff_Coordinate[1]));
        C[IS_RIGHT] = max(C[IS_RIGHT], (int) ceil(Buff_Coordinate[0]));
        D[IS_RIGHT] = max(D[IS_RIGHT], (int) ceil(Buff_Coordinate[1]));
    }
    int retImg_height = (C[IS_RIGHT] - A[IS_RIGHT] + 1), retImg_width = (D[IS_RIGHT] - B[IS_RIGHT] + 1);
    unsigned char *** ret_Img = AllocateImg_Uchar(retImg_height, retImg_width, channel);
    int * Buff_Idx = new int [3] {0, 0, 1};
    // Assign values to the return img
    for(int i = 0; i < retImg_height; ++i){
        for(int j = 0; j < retImg_width; ++j){
            Buff_Idx[0] = i + A[IS_RIGHT]; Buff_Idx[1] = j + B[IS_RIGHT];
            Buff_Coordinate = Mapping_DestiCart_to_ScrIdx(Index_to_Cartesian(Buff_Idx), "Q2");
            for(int k = 0; k < channel; ++k){
                ret_Img[k][i][j] = Interpolation_Bilinear(scr_Img[k], Buff_Coordinate, height, width);
            }
        }
    }
    delete[] scr_Img;
    if(IS_RIGHT){
        output_ppm("../Stitching/TranformedLeft", ret_Img, retImg_height, retImg_width, channel);
    }
    else{
        output_ppm("../Stitching/TranformedRight", ret_Img, retImg_height, retImg_width, channel);
    }
    return ret_Img;
}

unsigned char *** ee569_hw3_sol::Image_Stitching(vector<string> filenameList,
                                                 vector<int>    heightList,
                                                 vector<int>    widthList,
                                                 int channel, int centerImg_Idx, int ControlPoint_Num){
    // We are first extending to left
    unsigned char *** ret_Canvas = input_raw(filenameList[centerImg_Idx], heightList[centerImg_Idx],
                                             widthList[centerImg_Idx], 3, 0, "", true);
    int Canvas_height = heightList[centerImg_Idx], Canvas_width = widthList[centerImg_Idx];
    unsigned char *** Buff_ImgLeft, *** Buff_ImgRight, *** temp_Canvas;
    int *** Buff_Matching;
    // This is not image, we are allocating memory only
    float *** Buff_Cartesian = AllocateImg_Float(ControlPoint_Num, 3, 2);
    int Pnt_ImgLeft = centerImg_Idx - 1, Pnt_ImgRight = centerImg_Idx + 1;
    // Start Stitching
    while(Pnt_ImgLeft >= 0 || Pnt_ImgRight <= filenameList.size() - 1){
        // For Left
        if(Pnt_ImgLeft >= 0){
            Buff_ImgLeft = input_raw(filenameList[Pnt_ImgLeft], heightList[Pnt_ImgLeft], widthList[Pnt_ImgLeft],
                                     3, 0, "", true);
            Buff_Matching = FeatureMatching_FLANN(filenameList[Pnt_ImgLeft], filenameList[centerImg_Idx],
                                                  ControlPoint_Num, 0);
            // Transform Index => Cartesian Coordinate
            for(int i = 0; i < ControlPoint_Num; ++i){
                Buff_Cartesian[0][i] = Index_to_Cartesian(Buff_Matching[0][i]);
                Buff_Cartesian[1][i] = Index_to_Cartesian(Buff_Matching[1][i]);
            }
            // Calculate the Transition Matrix and its reverse, it's a class variable
            TransitionMx_Cal(Buff_Cartesian, ControlPoint_Num, "Q2");
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            // This is a test part of the reverse mapping by explicitly specifying the transition matrix
            // Transition Matrix Copied from Python script that takes 4 corners as controll point
            // Transition_Mx[0][0] = 2.0480264320910635; Transition_Mx[0][1] = 0.7614159526838122;
            // Transition_Mx[0][2] = -303.7682877685893; 
            // Transition_Mx[1][0] = 0.14988138001826457; Transition_Mx[1][1] = 2.219132721478587; 
            // Transition_Mx[1][2] = -444.6517348484922;
            // Transition_Mx[2][0] = 0.000459772601403341; Transition_Mx[2][1] = 0.002217162839317846;
            // Transition_Mx[2][2] = 1;
            // Transition_Mx_Rev = Matrix_ToolBox::Inverse_Cal(Transition_Mx, 3);
            //////////////////////////////////////////////////////////////////////////////////////////////////////
            // Linear Morph the Image
            Buff_ImgLeft = Image_LinearMorph_STEP(Buff_ImgLeft, heightList[Pnt_ImgLeft], widthList[Pnt_ImgLeft], 3, false);
        }
        // For Right
        if(Pnt_ImgRight <= filenameList.size() - 1){
            Buff_ImgRight = input_raw(filenameList[Pnt_ImgRight], heightList[Pnt_ImgRight], widthList[Pnt_ImgRight],
                                      3, 0, "", true);
            Buff_Matching = FeatureMatching_FLANN(filenameList[Pnt_ImgRight], filenameList[centerImg_Idx],
                                                  ControlPoint_Num, 1);
            // Transform Index => Cartesian Coordinate
            for(int i = 0; i < ControlPoint_Num; ++i){
                Buff_Cartesian[0][i] = Index_to_Cartesian(Buff_Matching[0][i]);
                Buff_Cartesian[1][i] = Index_to_Cartesian(Buff_Matching[1][i]);
            }
            // Calculate the Transition Matrix and its reverse, it's a class variable
            TransitionMx_Cal(Buff_Cartesian, ControlPoint_Num, "Q2");
            // Linear Morph the Image
            Buff_ImgRight = Image_LinearMorph_STEP(Buff_ImgRight, heightList[Pnt_ImgRight], widthList[Pnt_ImgRight], 3, true);
        }
        // Now we are Stitching the image together
        temp_Canvas = AllocateImg_Uchar(max(max(C[0], C[1]), Canvas_height) - min(min(A[0], A[1]), 0) + 1,
                                        max(max(D[0], D[1]), Canvas_width) - min(min(B[0], B[1]), 0) + 1,
                                        3);
        int shift_X, shift_Y;
        for(int k = 0; k < channel; ++k){
            // Add left
            shift_X = A[0] - min(min(A[0], A[1]), 0);
            shift_Y = B[0] - min(min(B[0], B[1]), 0);
            for(int i = 0; i < C[0] - A[0] + 1; ++i){
                for(int j = 0; j < D[0] - B[0] + 1; ++j){
                    temp_Canvas[k][i + shift_X][j + shift_Y] = Buff_ImgLeft[k][i][j];
                }
            }
            // Add right
            shift_X = A[1] - min(min(A[0], A[1]), 0);
            shift_Y = B[1] - min(min(B[0], B[1]), 0);
            for(int i = 0; i < C[1] - A[1] + 1; ++i){
                for(int j = 0; j < D[1] - B[1] + 1; ++j){
                    temp_Canvas[k][i + shift_X][j + shift_Y] = Buff_ImgRight[k][i][j];
                }
            }
            // Add Middle
            shift_X = - min(min(A[0], A[1]), 0);
            shift_Y = - min(min(B[0], B[1]), 0);
            for(int i = 0; i < Canvas_height; ++i){
                for(int j = 0; j < Canvas_width; ++j){
                    temp_Canvas[k][i + shift_X][j + shift_Y] = ret_Canvas[k][i][j];
                }
            }
        }
        Pnt_ImgLeft -= 1; Pnt_ImgRight += 1;
        ret_Canvas = temp_Canvas;
        Canvas_height = max(max(C[0], C[1]), Canvas_height) - min(min(A[0], A[1]), 0) + 1;
        Canvas_width = max(max(D[0], D[1]), Canvas_width) - min(min(B[0], B[1]), 0) + 1;
    }
    cout << "Image Stitching Completed" << endl;
    output_ppm("../Stitching/final", ret_Canvas, Canvas_height, Canvas_width, channel);
    return ret_Canvas;
}
//////////////////////////////////////////////////////////////////////////////
///////////////This is the section for Morphological processing///////////////
/////////////// All Mask Matching is done by bit-wise operations//////////////
//////////////////////////////////////////////////////////////////////////////
void ee569_hw3_sol::Image_Binarize(unsigned char ** scr_Img, int height, int width){
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            scr_Img[i][j] = 255 * (unsigned char) (scr_Img[i][j] >= Binarizing_Threshold);
        }
    }
}
void ee569_hw3_sol::Image_Reverse(unsigned char ** scr_Img, int height, int width){
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                scr_Img[i][j] = ~ scr_Img[i][j];
            }
        }
}
unsigned char ** ee569_hw3_sol::Image_Crop(unsigned char ** scr_Img, int height, int width, 
                                           int up , int down, int left, int right){
    unsigned char ** ret_Img = AllocateImg_Uchar(height - up - down, width - left - right, 1)[0];
    for(int i = up; i < height - down; ++i){
        for(int j = left; j < width - right; ++j){
            ret_Img[i - up][j - left] = scr_Img[i][j];
        }
    }
    return ret_Img;
}
unsigned char ** ee569_hw3_sol::Image_Copy(unsigned char ** scr_Img, int height, int width){
    unsigned char ** ret_Img = AllocateImg_Uchar(height, width, 1)[0];
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            ret_Img[i][j] = scr_Img[i][j];
        }
    }
    return ret_Img;
}
unsigned char *** ee569_hw3_sol::Image_Copy(unsigned char *** scr_Img, int height, int width, int channel){
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


vector<unsigned char> ee569_hw3_sol::Fetch_ConditionalMask_BASE(int BOND_NUM){
    vector<unsigned char> ret_Vec;
    switch(BOND_NUM){
        case 1:     // S
            ret_Vec = {0b01000000, 0b00010000, 0b00000100, 0b00000001};
            break;
        case 2:     // S
            ret_Vec = {0b10000000, 0b00100000, 0b00001000, 0b00000010};
            break;
        case 3:     // S
            ret_Vec = {0b11000000, 0b01100000, 0b00110000, 0b00011000,
                       0b00001100, 0b00000110, 0b00000011, 0b10000001};
            break;
        case 41:    // TK
            ret_Vec = {0b10100000, 0b00101000, 0b00001010, 0b10000010};
            break;
        case 42:    // STK
            ret_Vec = {0b11000001, 0b01110000, 0b00011100, 0b00000111};
            break;
        case 51:    // ST
            ret_Vec = {0b10110000, 0b10100001, 0b01101000, 0b11000010};
            break;
        case 52:    // ST
            ret_Vec = {0b11100000, 0b00111000, 0b00001110, 0b10000011};
            break;
        case 61:    // ST
            ret_Vec = {0b10110001, 0b01101100};
            break;
        case 62:    // STK
            ret_Vec = {0b11110000, 0b11100001, 0b01111000, 0b00111100,
                       0b00011110, 0b00001111, 0b10000111, 0b11000011};
            break;
        case 7:     // STK
            ret_Vec = {0b11110001, 0b01111100, 0b00011111, 0b11000111};
            break;
        case 8:     // STK
            ret_Vec = {0b11100011, 0b10001111, 0b11111000, 0b00111110};
            break;
        case 9:     // STK
            ret_Vec = {0b11110011, 0b01111110, 0b11111100, 0b11111001,
                       0b11100111, 0b00111111, 0b10011111, 0b11001111};
            break;
        case 10:    // STK
            ret_Vec = {0b11110111, 0b11111101, 0b01111111, 0b11011111};
            break;
        case 11:    // K
            ret_Vec = {0b11111011, 0b11111110, 0b10111111, 0b11101111};
            break;
        default:
            break;
    }
    return ret_Vec;
}
vector<unsigned char> ee569_hw3_sol::Fetch_ConditionalMask(unsigned char TYPE){
    vector<unsigned char> ret_Vec, Buff_Vec;
    vector<int> BOND_IDX;
    switch(TYPE){
        case SHRINKING:
            BOND_IDX = {1, 2, 3, 42, 51, 52, 61, 62, 7, 8, 9, 10};
            break;
        case THINNING:
            BOND_IDX = {41, 42, 51, 52, 61, 62, 7, 8, 9, 10};
            break;
        case SKELETONIZING:
            BOND_IDX = {41, 42, 62, 7, 8, 9,  10, 11};
            break;
        default:
            break;
    }
    for(int i = 0; i < BOND_IDX.size(); ++i){
        Buff_Vec = Fetch_ConditionalMask_BASE(BOND_IDX[i]);
        ret_Vec.insert(ret_Vec.end(), Buff_Vec.begin(), Buff_Vec.end());
    }
    return ret_Vec;
}
vector<array<unsigned char, 3>> ee569_hw3_sol::Fetch_UnconditionalMask(bool FOR_ST){
    bool FOR_K = !FOR_ST;
    vector<array<unsigned char, 3>> ret_Vec;
    // Spur
    ret_Vec.push_back({0b01000000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    ret_Vec.push_back({0b00010000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    if(FOR_K){
        ret_Vec.push_back({0b00000100, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00000001, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    }
    // Single 4-connections
    ret_Vec.push_back({0b00000010, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    ret_Vec.push_back({0b10000000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    if(FOR_K){
        ret_Vec.push_back({0b00100000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00001000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    }
    // L Cluster
    if(FOR_ST){
        ret_Vec.push_back({0b11000000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b01100000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00110000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00011000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00001100, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00000110, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00000011, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10000001, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    }
    // L Corner
    if(FOR_K){
        ret_Vec.push_back({0b10100000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00101000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00001010, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10000010, MASK_FOR_ONLY01, MASK_FOR_NOABC});
    }
    if(FOR_ST){
        // 4-Connected Offset
        ret_Vec.push_back({0b01101000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10110000, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10100001, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        ret_Vec.push_back({0b11000010, MASK_FOR_ONLY01, MASK_FOR_NOABC});
        // Spur Corner Cluster
        ret_Vec.push_back({0b01000100, 0b01011111, 0b10100000});
        ret_Vec.push_back({0b00010001, 0b11010111, 0b00101000});
        ret_Vec.push_back({0b01000100, 0b11110101, 0b00001010});
        ret_Vec.push_back({0b00010001, 0b01111101, 0b10000010});
    }
    // Corner Cluster
    
    // THIS IS THE ORIGINAL MASK
    if(Mask_UseOriginal){
        ret_Vec.push_back({0b00111000, 0b00111000, MASK_FOR_NOABC});
        if(FOR_K){
            ret_Vec.push_back({0b10000011, 0b10000011, MASK_FOR_NOABC});
        }
    }
    // THIS SOLVES DISCONNECTION
    else if(!Mask_UseOriginal){    
        ret_Vec.push_back({0b00001110, 0b00001110, MASK_FOR_NOABC});
        if(FOR_K){
            ret_Vec.push_back({0b11100000, 0b11100000, MASK_FOR_NOABC});
        }
    }

    // Tee Branch
    if(FOR_ST){
        ret_Vec.push_back({0b10101000, 0b11101011, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10101000, 0b10111110, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10001010, 0b10111110, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10001010, 0b11101011, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00101010, 0b10101111, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00101010, 0b11111010, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10100010, 0b11111010, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10100010, 0b10101111, MASK_FOR_NOABC});
    }
    if(FOR_K){
        ret_Vec.push_back({0b10101000, 0b10101000, MASK_FOR_NOABC});
        ret_Vec.push_back({0b00101010, 0b00101010, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10001010, 0b10001010, MASK_FOR_NOABC});
        ret_Vec.push_back({0b10100010, 0b10100010, MASK_FOR_NOABC});
    }
    // Vee Branch
    ret_Vec.push_back({0b01010000, 0b01010000, 0b00000111});
    ret_Vec.push_back({0b00010100, 0b00010100, 0b11000001});
    ret_Vec.push_back({0b00000101, 0b00000101, 0b01110000});
    ret_Vec.push_back({0b01000001, 0b01000001, 0b00011100});
    // Diagonal Branch
    ret_Vec.push_back({0b10100100, 0b11101110, MASK_FOR_NOABC});
    ret_Vec.push_back({0b00101001, 0b10111011, MASK_FOR_NOABC});
    ret_Vec.push_back({0b01001010, 0b11101110, MASK_FOR_NOABC});
    ret_Vec.push_back({0b10010010, 0b10111011, MASK_FOR_NOABC});
    return ret_Vec;
}
bool ee569_hw3_sol::Match_ConditionalMask(vector<unsigned char> Mask_Pool, unsigned char Byte_Input){
    for(int i = 0; i < Mask_Pool.size(); ++i){
        if((Byte_Input ^ Mask_Pool[i]) == 0x00)
            return true;
    }
    return false;
}
bool ee569_hw3_sol::Match_UnconditionalMask(vector<array<unsigned char, 3>> Mask_Pool, unsigned char Byte_Input){
    bool Buff_Bool1, Buff_Bool2;
    unsigned char M01, M, MA;
    for(int i = 0; i < Mask_Pool.size(); ++i){
        M01 = Mask_Pool[i][0]; M = Mask_Pool[i][1]; MA = Mask_Pool[i][2];
        Buff_Bool1 = (((Byte_Input & M) ^ M01) == 0x00);
        if(MA == MASK_FOR_NOABC){
            Buff_Bool2 = true;
        }
        else{
            Buff_Bool2 = ((Byte_Input & MA) != 0x00);
        }

        if(Buff_Bool1 && Buff_Bool2) 
            return true;
    }
    return false;
}
unsigned char ee569_hw3_sol::Reorder_Neighborhood(unsigned char ** scr_Img, int i, int j){
    // We make sure boundary is not crossed from the outside of this function
    unsigned char Val_Match = 0xFF;
    unsigned char ret_Byte = ((unsigned char)(scr_Img[i][j + 1] == Val_Match)     << 7) +
                             ((unsigned char)(scr_Img[i - 1][j + 1] == Val_Match) << 6) +
                             ((unsigned char)(scr_Img[i - 1][j] == Val_Match)     << 5) +
                             ((unsigned char)(scr_Img[i - 1][j - 1] == Val_Match) << 4) +
                             ((unsigned char)(scr_Img[i][j - 1] == Val_Match)     << 3) +
                             ((unsigned char)(scr_Img[i + 1][j - 1] == Val_Match) << 2) + 
                             ((unsigned char)(scr_Img[i + 1][j] == Val_Match)     << 1) +
                             (unsigned char)(scr_Img[i + 1][j + 1] == Val_Match);
    return ret_Byte;
}
void ee569_hw3_sol::Morph_OneIter(unsigned char ** scr_Img, int height, int width, int maxIter, unsigned char TYPE){
    unsigned char ** Buff_Mask, Buff_Byte;
    unsigned char ** Buff_Img;
    vector<unsigned char> CondiMask_Pool = Fetch_ConditionalMask(TYPE);
    vector< array<unsigned char, 3> > UncondiMask_Pool = Fetch_UnconditionalMask(TYPE != SKELETONIZING);
    for(int iter = 0; iter < maxIter; ++iter){
        Buff_Mask = AllocateImg_Uchar(height + 4, width + 4, 1)[0];
        // Conditional Masking
        for(int i = 1; i < height + 3; ++i){
            for(int j = 1; j < width + 3; ++j){
                if(scr_Img[i][j] == 0x00)
                    continue;
                Buff_Byte = Reorder_Neighborhood(scr_Img, i, j);
                Buff_Mask[i][j] = 255 * (unsigned char) Match_ConditionalMask(CondiMask_Pool, Buff_Byte);
            }
        }
        for(int i = 2; i < height + 2; ++i){
            for(int j = 2; j < width + 2; ++j){
                if(scr_Img[i][j] == 0x00 || Buff_Mask[i][j] == 0x00)
                    continue;
                Buff_Byte = Reorder_Neighborhood(Buff_Mask, i, j);
                if(!Match_UnconditionalMask(UncondiMask_Pool, Buff_Byte))
                    scr_Img[i][j] = 0x00;
            }
        }
        delete[] Buff_Mask;
    }
}
unsigned char *** ee569_hw3_sol::Morph_DO(unsigned char *** scr_Img, int height, int width, int max_Iter,
                                          unsigned char TYPE, string outputName){
    // Adding Test shapes to "fan"
    if(outputName == "Morph_progress/fan"){
        for(int i = 0; i < height + 4; ++i){
            for(int j = 0; j < width + 4; ++j){
                if((20 <= i && i <= 120) && (20 <= j && j <= 200))
                    scr_Img[0][i][j] = 0xFF;
                else if((pow((float(i) - 500), 2) / 4 + pow((float(j) - 500), 2) / 10) < 250)
                    scr_Img[0][i][j] = 0xFF;
            }
        }
    }
    Morph_OneIter(scr_Img[0], height, width, max_Iter, TYPE);
    output_pgm("../" + outputName + to_string(TYPE) + "_" + to_string(max_Iter),
                scr_Img, height + 4, width + 4, 1);
    return scr_Img;
}
void ee569_hw3_sol::TransitionMx_Cal(float theta, float i, float j){
    Transition_Mx[0][0] = cos(theta); Transition_Mx[0][1] = -sin(theta);
    Transition_Mx[1][0] = sin(theta); Transition_Mx[1][1] = cos(theta);
    Transition_Mx[0][2] = (1 - cos(theta)) * i + sin(theta) * j;
    Transition_Mx[1][2] = -sin(theta) * i + (1 - cos(theta)) * j;
    Transition_Mx[2][0] = 0; Transition_Mx[2][1] = 0; Transition_Mx[2][2] = 1;
    Transition_Mx_Rev = Matrix_ToolBox::Inverse_Cal(Transition_Mx, 3);
}
void ee569_hw3_sol::MorphologicalProcess_Basic(int lowIter, int highIter, int stepIter){
    if(Print_TestMessage){
        cout << "------------THIS IS A TEST FOR REORDERING_NEIGHBORHOOD()------------" << endl;
        unsigned char ** testImg = AllocateImg_Uchar(3, 3, 1)[0];
        testImg[0][0] = 255; testImg[0][1] = 255; testImg[0][2] = 255;
        testImg[1][0] = 255; testImg[1][1] = 255; testImg[1][2] = 0;
        testImg[2][0] = 255; testImg[2][1] = 255; testImg[2][2] = 255;
        cout << (int) Reorder_Neighborhood(testImg, 1, 1) << endl;
        cout << "---------------------------TEST ENDS--------------------------------" << endl;
    }
    vector<string>          filenameList = {"fan", "cup", "maze"};
    vector<int>             heightList = {558, 356, 558};
    vector<int>             widthList = {558, 315, 558};
    vector<unsigned char>   opList = {THINNING, SHRINKING, SKELETONIZING};
    unsigned char *** scr_Img;
    for(int i = 0; i < filenameList.size(); ++i){
        for(int op = 0; op < opList.size(); ++op){
            for(int iter = lowIter; iter <= highIter; iter += stepIter){
                scr_Img = input_raw("../" + filenameList[i], heightList[i], widthList[i], 1, 2, "", true);
                Image_Binarize(scr_Img[0], heightList[i] + 4, widthList[i] + 4);
                Morph_DO(scr_Img, heightList[i], widthList[i], iter, opList[op], "Morph_progress/" + filenameList[i]);
            }
        }
    }
}
void ee569_hw3_sol::MorphologicalProcess_CountStars(string filename, int height, int width, int channel){
    // First we use thrinking method to count the stars
    Binarizing_Threshold = 50;
    int cnt = 0;
    unsigned char *** scr_Img = input_raw(filename, height, width, channel, 2, "", true);
    Image_Binarize(scr_Img[0], height, width);
    // Count the stars first
    scr_Img = Morph_DO(scr_Img, height, width, 50, SHRINKING, "Stars_Counting/STAR");
    for(int i = 2; i < height + 2; ++i){
        for(int j = 2; j < width + 2; ++j){
            if(scr_Img[0][i][j] == 0xFF)
                cnt += 1;
        }
    }
    cout << "The total number of stars counted by shrinking is " << cnt << endl;
    delete[] scr_Img;
    // Second, we use depth first search
    scr_Img = input_raw(filename, height, width, channel, 0, "", true);
    cnt = 0;
    Image_Binarize(scr_Img[0], height, width);
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            if(scr_Img[0][i][j] == 0xFF){
                cnt += 1;
                DFS(scr_Img[0], i, j, height, width);
            }
        }
    }
    cout << "The total number of stars counted by DFS is " << cnt << endl;
    // cout << "----------------------THIS IS A REMINDER----------------------" << endl;
    // cout << "----INCORRECTNESS OF SHRINKING IS THE REASON FOR DIFFERENCE---" << endl;
    // cout << "---------------------------REMINDER ENDS----------------------" << endl;
    delete[] scr_Img;
    // Thirdly, we are counting the star size, using modified DFS
    vector <int> sizePool = {};
    scr_Img = input_raw(filename, height, width, channel, 0, "", true);
    Image_Binarize(scr_Img[0], height, width);
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            if(scr_Img[0][i][j] == 0xFF)
                sizePool.push_back(DFS(scr_Img[0], i, j, height, width));
        }
    }
    cout << "The distribution of the star size is:";
    for(int i = 0; i < sizePool.size(); ++i){
        if(i % 5 == 0)
            cout << endl;
        cout << sizePool[i] << " ";

    }
    cout << endl;
    delete[] scr_Img;
}
// We are overloading this function to serve two slightly different purposes
int ee569_hw3_sol::DFS(unsigned char ** scr_Img, int i, int j, int height, int width, unsigned char foodVal){
    if(i < 0 || i == height || j < 0 || j == width)
        return 1;
    else if(scr_Img[i][j] == 0x00)
        return 0;
    bool is_gray = (scr_Img[i][j] == foodVal);
    scr_Img[i][j] = 0x00;
    return  (int) is_gray + DFS(scr_Img, i + 1, j + 1, height, width, foodVal) + 
                            DFS(scr_Img, i + 1, j - 1, height, width, foodVal) + 
                            DFS(scr_Img, i + 1, j, height, width, foodVal) + 
                            DFS(scr_Img, i, j - 1, height, width, foodVal) + 
                            DFS(scr_Img, i, j + 1, height, width, foodVal) + 
                            DFS(scr_Img, i - 1, j + 1, height, width, foodVal) + 
                            DFS(scr_Img, i - 1, j - 1, height, width, foodVal) + 
                            DFS(scr_Img, i - 1, j, height, width, foodVal);
}
int ee569_hw3_sol::DFS(unsigned char ** scr_Img, int i, int j, int height, int width){
    if(i < 0 || i == height || j < 0 || j == width || scr_Img[i][j] == 0x00)
        return 0;
    scr_Img[i][j] = 0x00;
    return  1 + DFS(scr_Img, i + 1, j + 1, height, width) + 
                DFS(scr_Img, i + 1, j - 1, height, width) + 
                DFS(scr_Img, i + 1, j, height, width) + 
                DFS(scr_Img, i, j - 1, height, width) + 
                DFS(scr_Img, i, j + 1, height, width) + 
                DFS(scr_Img, i - 1, j + 1, height, width) + 
                DFS(scr_Img, i - 1, j - 1, height, width) + 
                DFS(scr_Img, i - 1, j, height, width);
}
void ee569_hw3_sol::MorphologicalProcess_PCBanalysis(string filename, int height, int width, int channel){
    Binarizing_Threshold = 56;
    // Preprocessing, outputing some intermediate images
    unsigned char *** scr_Img = input_raw("../PCB", height, width, channel, 2, "", true);
    Image_Binarize(scr_Img[0], height + 4, width + 4);
    output_pgm("../PCB_analysis/Binarized", scr_Img, height + 4, width + 4, channel);
    Image_Reverse(scr_Img[0], height + 4, width + 4);
    output_pgm("../PCB_analysis/Rev", scr_Img, height + 4, width + 4, channel);
    // We are now counting the holes
    vector<int *> holePool = {};
    int * Buff_holePnt;
    unsigned char Buff_Byte;
    scr_Img = input_raw("../PCB", height, width, channel, 2, "", true);
    Image_Binarize(scr_Img[0], height + 4, width + 4);
    Morph_DO(scr_Img, height, width, 50, SHRINKING, "PCB_analysis/pcb_ori");
    scr_Img[0] = Image_Crop(scr_Img[0], height + 4, width + 4, 1, 1, 1, 1);
    output_pgm("../PCB_analysis/Cropped_Hole", scr_Img, height, width, channel);
    for(int i = 1; i < height; ++i){
        for(int j = 1; j < width; ++j){
            if(scr_Img[0][i][j] == 0x00)
                continue;
            Buff_Byte = Reorder_Neighborhood(scr_Img[0], i, j);
            if((Buff_Byte ^ 0x00) == 0x00){
                Buff_holePnt = new int [2] {i - 1, j - 1};
                holePool.push_back(Buff_holePnt);
            }
        }
    }
    delete[] scr_Img;
    cout << "The number of holes is " << holePool.size() << endl;
    // We are now counting the wires
    scr_Img = input_raw("../PCB", height, width, channel, 2, "", true);
    int row, col, cnt = 0;
    Image_Binarize(scr_Img[0], height + 4, width + 4);
    Image_Reverse(scr_Img[0], height + 4, width + 4);
    Morph_DO(scr_Img, height, width, 50, SHRINKING, "PCB_analysis/pcb_rev");
    scr_Img[0] = Image_Crop(scr_Img[0], height + 4, width + 4, 2, 2, 2, 2);
    // Mark the first white pixel above hole center gray
    for(auto holePnt: holePool){
        row = holePnt[0]; col = holePnt[1];
        while(scr_Img[0][row][col] == 0x00){
            col -= 1;
        }
        scr_Img[0][row][col] = 0x80;
    }
    // Do DFS
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            if(scr_Img[0][i][j] == 0x80 && (DFS(scr_Img[0], i, j, height, width, 0x80) > 1))
                cnt += 1;
        }
    }
    cout << "The number of wires is " << cnt << endl;
}
void ee569_hw3_sol::MorphologicalProcess_DefeatDetection(string filename, int height, int width, int channel){
    unsigned char *** scr_Img = input_raw("../GearTooth", 250, 250, 1, 2, "", true);
    // First we have to determine the location of the center
    Binarizing_Threshold = 128;
    unsigned char Buff_Byte;
    float center_i = 0., center_j = 0.;
    Image_Binarize(scr_Img[0], height + 4, width + 4);
    Image_Reverse(scr_Img[0], height + 4, width + 4);
    Morph_DO(scr_Img, height, width, 50, SHRINKING, "DefectDetection/rev");
    for(int i = 1; i < height + 3; ++i){
        for(int j = 1; j < width + 3; ++j){
            if(scr_Img[0][i][j] == 0x00)
                continue;
            Buff_Byte = Reorder_Neighborhood(scr_Img[0], i, j);
            if((Buff_Byte ^ 0x00) == 0x00){
                center_i += (i - 2);
                center_j += (j - 2);
            }
        }
    }
    center_i /= 4; center_j /= 4;
    center_i += 1./2; center_j += 1./2;
    // Begin Rotating, total iteration = 12
    // In this case, only one iteration is needed
    scr_Img = input_raw("../GearTooth", 250, 250, 1, 0, "", true);
    int *** Buff_Img = AllocateImg_Int(height, width, channel);
    int * Buff_Idx = new int [2];
    float * Buff_Coordinate;
    float weight_Ori = 24;
    // add weight to the original
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            Buff_Img[0][i][j] += weight_Ori * (int) scr_Img[0][i][j];
        }
    }
    for(int iter = 1; iter <= 12; ++iter){
        // Calculate the transformation matrix
        TransitionMx_Cal((float) iter * M_PI / 6, center_i, center_j);
        // Rotate the image & add to Buff_Img
        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                Buff_Idx[0] = i; Buff_Idx[1] = j; Buff_Idx[2] = 1.;
                Buff_Coordinate = Mapping_DestiCart_to_ScrIdx(Index_to_Cartesian(Buff_Idx), "Q2");
                Buff_Img[0][i][j] += (int) Interpolation_Bilinear(scr_Img[0], Buff_Coordinate, height, width);
            }
        }
    }
    // Intermediate Step Visualization & Recover Black holes
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            scr_Img[0][i][j] = Buff_Img[0][i][j] / (12 + weight_Ori);
        }
    }
    output_pgm("../DefectDetection/Rotated", scr_Img, height, width, channel);
    Binarizing_Threshold = 60;
    Image_Binarize(scr_Img[0], height, width);
    output_pgm("../DefectDetection/RotatedBinarized", scr_Img, height, width, channel);
    delete[] Buff_Img;
    // We want to clear the isolated dots, who doesn't have any 4-connections
    for(int i = 0; i < height; ++i){
        for(int j = 0; j < width; ++j){
            if(scr_Img[0][i][j] == 0x00)
                continue;
            Buff_Byte = Reorder_Neighborhood(scr_Img[0], i, j);
            if((Buff_Byte & 0b10101010) == 0x00)
                scr_Img[0][i][j] = 0x00;
        }
    }
    output_pgm("../DefectDetection/Denoised", scr_Img, height, width, channel);
    // We are generating the final image
    unsigned char *** ori_Img = input_raw("../GearTooth", 250, 250, 1, 0, "", true);
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; ++j){
            if(ori_Img[0][i][j] == 0x00 && scr_Img[0][i][j] == 0xFF)
                scr_Img[0][i][j] = 0x80;
        }
    }
    output_pgm("../DefectDetection/final", scr_Img, height, width, channel);
}
//////////////////////////////////////////////////////////////////////////////
void ee569_hw3_sol::READ_ALL_RAW(){
    output_ppm("../hedwig_ori", input_raw("../hedwig", 512, 512, 3, 0, "", true), 512, 512, 3);
    output_ppm("../raccoon_ori", input_raw("../raccoon", 512, 512, 3, 0, "", true), 512, 512, 3);
    output_ppm("../bb8_ori", input_raw("../bb8", 512, 512, 3, 0, "", true), 512, 512, 3);
    output_ppm("../left_ori", input_raw("../left", 720, 480, 3, 0, "", true), 720, 480, 3);
    output_ppm("../middle_ori", input_raw("../middle", 720, 480, 3, 0, "", true), 720, 480, 3);
    output_ppm("../right_ori", input_raw("../right", 720, 480, 3, 0, "", true), 720, 480, 3);
    output_pgm("../fan", input_raw("../fan", 558, 558, 1, 0, "", true), 558, 558, 1);
    output_pgm("../cup", input_raw("../cup", 356, 315, 1, 0, "", true), 356, 315, 1);
    output_pgm("../maze", input_raw("../maze", 558, 558, 1, 0, "", true), 558, 558, 1);
    output_pgm("../stars", input_raw("../stars", 480, 640, 1, 0, "", true), 480, 640, 1);
    output_pgm("../PCB", input_raw("../PCB", 239, 372, 1, 0, "", true), 239, 372, 1);
    output_pgm("../GearTooth", input_raw("../GearTooth", 250, 250, 1, 0, "", true), 250, 250, 1);
}
void ee569_hw3_sol::pxm_TO_RAW_ALL(){
    vector<string> filenameList = { "bb8_RevWarped", "bb8_Warped",
                                    "hedwig_RevWarped", "hedwig_Warped",
                                    "raccoon_RevWarped", "raccoon_Warped",
                                    // "matches0Selected", "matches1Selected",
                                    "StitchFinal",
                                    "cupT", "cupS", "cupK",
                                    "fanT", "fanS", "fanK",
                                    "mazeT", "mazeS", "mazeK",
                                    "pcb_OriS50", "pcb_RevS50", 
                                    "StarS50",
                                    "GearFinal", "Denoised"
                                    };
    vector<array<int, 3>> dimsList = {{512, 512, 3}, {512, 512, 3},
                                      {512, 512, 3}, {512, 512, 3},
                                      {512, 512, 3}, {512, 512, 3},
                                      // {720, 960, 3}, {720, 960, 3},
                                      {1160, 1166, 3},
                                      {360, 319, 1}, {360, 319, 1}, {360, 319, 1}, 
                                      {562, 562, 1}, {562, 562, 1}, {562, 562, 1}, 
                                      {562, 562, 1}, {562, 562, 1}, {562, 562, 1}, 
                                      {243, 376, 1}, {243, 376, 1}, 
                                      {484, 644, 1},
                                      {250, 250, 1}, {250, 250, 1}
                                    };

    for(int i = 0; i < filenameList.size(); ++i){
        if(dimsList[i][2] == 3)
            output_raw("../AllRaw/" + filenameList[i], input_ppm("../AllRaw/" + filenameList[i]),
                    dimsList[i][0], dimsList[i][1], dimsList[i][2]);
        else if(dimsList[i][2] == 1)
            output_raw("../AllRaw/" + filenameList[i], input_pgm("../AllRaw/" + filenameList[i]),
                    dimsList[i][0], dimsList[i][1], dimsList[i][2]);
    }
}