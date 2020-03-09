//
// Created by Administrator on 2020/3/3.
//

# include "Matrix_ToolBox.hpp"

# include <stdio.h>
# include <iostream>
# include <vector>

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
vector<vector<float>> Matrix_ToolBox::Allocate_Mx_Vec(int M, int N){
    vector<vector<float>> ret_Mx;
    vector<float>         Buff_Mx;
    for(int i = 0; i < M; ++i){
        Buff_Mx.clear();
        for(int j = 0; j < N; ++j){
            Buff_Mx.push_back(0.);
        }
        ret_Mx.push_back(Buff_Mx);
    }
    return ret_Mx;
}
vector<vector<int>> Matrix_ToolBox::Allocate_Mx_Vec_Int(int M, int N){
    vector<vector<int>> ret_Mx;
    vector<int>         Buff_Mx;
    for(int i = 0; i < M; ++i){
        Buff_Mx.clear();
        for(int j = 0; j < N; ++j){
            Buff_Mx.push_back(0);
        }
        ret_Mx.push_back(Buff_Mx);
    }
    return ret_Mx;
}
vector<float> Matrix_ToolBox::Allocate_Vec_Vec(int M){
    vector<float> ret_Vec;
    for(int i = 0; i < M; ++i){
        ret_Vec.push_back(0.);
    }
    return ret_Vec;
}
vector<int> Matrix_ToolBox::Allocate_Vec_Vec_Int(int M){
    vector<int> ret_Vec;
    for(int i = 0; i < M; ++i){
        ret_Vec.push_back(0.);
    }
    return ret_Vec;
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
float ** Matrix_ToolBox::Matrix_Mul(float ** left_Mx, float ** right_Mx, int dim[4]){
    int left_M = dim[0], left_N = dim[1], right_M = dim[2], right_N = dim[3];
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
float * Matrix_ToolBox::Matrix_Mul(float * left_Vec, float ** right_Mx, int dim[4]){
    int left_M = dim[0], left_N = dim[1], right_M = dim[2], right_N = dim[3];
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
float * Matrix_ToolBox::Matrix_Mul(float ** left_Mx, float * right_Vec, int dim[4]){
    int left_M = dim[0], left_N = dim[1], right_M = dim[2], right_N = dim[3];
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
vector<vector<float>> Matrix_ToolBox::Matrix_Mul(vector<vector<float>> leftMx, vector<vector<float>> rightMx){
    int leftM = leftMx.size(), leftN = leftMx[0].size(), rightM = rightMx.size(), rightN = rightMx[0].size();
    if(leftN != rightM)
        throw "The dimensions doesn't match";
    vector<vector<float>> ret_Mx;
    vector<float> Buff_Vec;
    float buff_val;
    for(int i = 0; i < leftM; ++i){
        Buff_Vec.clear();
        for(int j = 0; j < rightN; ++j){
            buff_val = 0.;
            for(int k = 0; k < leftN; ++k){
                buff_val += leftMx[i][k] * rightMx[k][j];
            }
            Buff_Vec.push_back(buff_val);
        }
        ret_Mx.push_back(Buff_Vec);
    }
    return ret_Mx;
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
// Tensor Product
float ** Matrix_ToolBox::Tensor_Product(float * leftVec, float * rightVec, int dim[2]){
    float ** ret_Mx = Allocate_Mx(dim[0], dim[1]);
    for(int i = 0; i < dim[0]; ++i){
        for(int j = 0; j < dim[1]; ++j){
            ret_Mx[i][j] = leftVec[i] * rightVec[j];
        }
    }
    return ret_Mx;
}