//
// Created by Administrator on 2020/3/3.
//

#ifndef MATRIX_TOOLBOX_HPP
#define MATRIX_TOOLBOX_HPP

# include <stdio.h>
# include <iostream>
# include <vector>

class Matrix_ToolBox{
    public:
        Matrix_ToolBox(){}
        ~ Matrix_ToolBox(){}
        static double    Determinant_Cal(double ** scr_Mx, int n);
        static float **  Inverse_Cal(double ** scr_Mx, int n);
        static float **  Inverse_Cal(float ** scr_Mx, int n);
        static float **  Matrix_Mul(float ** left_Mx, float ** right_Mx, int dim[4]);
        static float  *  Matrix_Mul(float * left_Vec, float ** right_Mx, int dims[4]);
        static float  *  Matrix_Mul(float ** left_Mx, float * right_Vec, int dims[4]);
        static std::vector<std::vector<float>> Matrix_Mul(std::vector<std::vector<float>> leftMx,
                                                          std::vector<std::vector<float>> rightMx);
        static float **  Matrix_Transpose(float ** scr_Mx, int scr_M, int scr_N);
        static float **  Allocate_Mx(int M, int N);
        static float **  Allocate_Mx(int N);
        static double ** Allocate_Mx_Double(int N);
        static double ** Allocate_Mx_Double(int M, int N);
        static std::vector<std::vector<float>> Allocate_Mx_Vec(int M, int N);
        static std::vector<std::vector<int>> Allocate_Mx_Vec_Int(int M, int N);
        static std::vector<float> Allocate_Vec_Vec(int M);
        static std::vector<int> Allocate_Vec_Vec_Int(int M);
        static float **  Tensor_Product(float * leftVec, float * rightVec, int dim[2]);
    private:
        static void Adjoint_Cal(double ** scr_Mx, double ** adj_Mx, int n);
        static void Cofactor_Cal(double ** scr_Mx, double ** temp_Mx, int p, int q, int n);

};
#endif //EE569HW4_MATRIX_TOOLBOX_HPP
