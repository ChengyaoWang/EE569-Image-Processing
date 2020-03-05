void ee569_hw3_sol::TransitionMx_Cal(float *** matching_Mx, int ControllPnt_Num, string TYPE){
    // TYPE1: Averaging A & B Method, TYPE2: Using Pseudo Inverse Method
    float ** Xin = Matrix_ToolBox::Matrix_Transpose(matching_Mx[0], ControllPnt_Num, 3);
    float ** Xout = Matrix_ToolBox::Matrix_Transpose(matching_Mx[1], ControllPnt_Num, 3);
    int * dims;
    if(TYPE == "TYPE1"){
        int GroupNum_Total = ControllPnt_Num / 4;
        float ** A = Matrix_ToolBox::Allocate_Mx(3), ** B = Matrix_ToolBox::Allocate_Mx(3);
        float ** Buff_MxSample = Matrix_ToolBox::Allocate_Mx(3), ** Buff_MxInv = Matrix_ToolBox::Allocate_Mx(3);
        float * Buff_PntSample = new float [3], * Buff_Coef;
        dims = new int [4] {3, 3, 3, 1};
        for(int group_pnt = 0; group_pnt <= GroupNum_Total; ++group_pnt){
            cout << group_pnt << endl;
            // Compute A
            for(int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Buff_MxSample[i][j] = Xin[i][4 * group_pnt + j];
                }
                Buff_PntSample[i] = Xin[i][4 * group_pnt + 3];
            }
            Matrix_ToolBox::Inverse_Cal(Buff_MxSample, Buff_MxInv, 3);
            Buff_Coef = Matrix_ToolBox::Matrix_Mul(Buff_MxInv, Buff_PntSample, dims);
            // Hardmard Product + Add to A
            for(int i = 0; i < 3; ++i){
                for(int j = 0; j < 3; ++j){
                    A[i][j] += Buff_MxSample[i][j] * Buff_Coef[j];
                }
            }
            // Compute B
            for(int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    Buff_MxSample[i][j] = Xout[i][4 * group_pnt + j];
                }
                Buff_PntSample[i] = Xout[i][4 * group_pnt + 3];
            }
            Matrix_ToolBox::Inverse_Cal(Buff_MxSample, Buff_MxInv, 3);
            Buff_Coef = Matrix_ToolBox::Matrix_Mul(Buff_MxInv, Buff_PntSample, dims);
            // Hardmard Product + Add to B
            for(int i = 0; i < 3; ++i){
                for(int j = 0; j < 3; ++j){
                    B[i][j] += Buff_MxSample[i][j] * Buff_Coef[j];
                }
            }
        }
        // Compute C
        dims = new int [4] {3, 3, 3, 3};
        Matrix_ToolBox::Inverse_Cal(A, Buff_MxInv, 3);
        Transition_Mx = Matrix_ToolBox::Matrix_Mul(B, Buff_MxInv, dims);
    }
    else if(TYPE == "TYPE2"){
        float ** A, ** B;
        float ** Buff_MxSample = Matrix_ToolBox::Allocate_Mx(3, ControllPnt_Num - 1);
        float ** Buff_MxInv = Matrix_ToolBox::Allocate_Mx(3);
        float * Buff_PntSample = new float [3];
        float * Buff_Coef = new float [ControllPnt_Num - 1];
        dims = new int [4] {3, ControllPnt_Num - 1, ControllPnt_Num - 1, 3};
        // Compute A
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < ControllPnt_Num - 1; ++j){
                Buff_MxSample[i][j] = Xin[i][j];
            }
            Buff_PntSample[i] = Xin[i][ControllPnt_Num - 1];
        }
        A = Matrix_ToolBox::Matrix_Mul(Buff_MxSample, Matrix_ToolBox::Matrix_Transpose(Buff_MxSample, 3, ControllPnt_Num - 1), dims);
        Matrix_ToolBox::Inverse_Cal(A, Buff_MxInv, 3);
        delete[] dims; dims = new int [4] {ControllPnt_Num - 1, 3, 3, 3};
        A = Matrix_ToolBox::Matrix_Mul(Matrix_ToolBox::Matrix_Transpose(Buff_MxSample, 3, ControllPnt_Num - 1), A, dims);
        delete[] dims; dims = new int [4] {ControllPnt_Num - 1, 3, 3, 1};
        Buff_Coef = Matrix_ToolBox::Matrix_Mul(A, Buff_PntSample, dims);
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < ControllPnt_Num - 1; ++j){
                A[i][j] = Buff_MxSample[i][j] * Buff_Coef[j];
            }
        }
        // Compute B
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < ControllPnt_Num - 1; ++j){
                Buff_MxSample[i][j] = Xout[i][j];
            }
            Buff_PntSample[i] = Xout[i][ControllPnt_Num - 1];
        }
        delete[] dims; dims = new int [4] {3, ControllPnt_Num - 1, ControllPnt_Num - 1, 3};
        B = Matrix_ToolBox::Matrix_Mul(Buff_MxSample, Matrix_ToolBox::Matrix_Transpose(Buff_MxSample, 3, ControllPnt_Num - 1), dims);
        Matrix_ToolBox::Inverse_Cal(B, Buff_MxInv, 3);
        delete[] dims; dims = new int [4] {ControllPnt_Num - 1, 3, 3, 3};
        B = Matrix_ToolBox::Matrix_Mul(Matrix_ToolBox::Matrix_Transpose(Buff_MxSample, 3, ControllPnt_Num - 1), B, dims);
        delete[] dims; dims = new int [4] {ControllPnt_Num - 1, 3, 3, 1};
        Buff_Coef = Matrix_ToolBox::Matrix_Mul(B, Buff_PntSample, dims);
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < ControllPnt_Num - 1; ++j){
                B[i][j] = Buff_MxSample[i][j] * Buff_Coef[j];
            }
        }
        // Compute C
        delete[] dims; dims = new int [4] {3, ControllPnt_Num - 1, ControllPnt_Num - 1, 3};
        B = Matrix_ToolBox::Matrix_Mul(B, Matrix_ToolBox::Matrix_Transpose(A, 3, ControllPnt_Num - 1), dims);
        A = Matrix_ToolBox::Matrix_Mul(B, Matrix_ToolBox::Matrix_Transpose(A, 3, ControllPnt_Num - 1), dims);
        Matrix_ToolBox::Inverse_Cal(A, Buff_MxInv, 3);
        delete[] dims; dims = new int [4] {3, 3, 3, 3};
        Transition_Mx = Matrix_ToolBox::Matrix_Mul(B, Buff_MxInv, dims);
    }
    Matrix_ToolBox::Inverse_Cal(Transition_Mx, Transition_Mx_Rev, 3);
    ////////////////////////This Method is deprecated because of in-correctness////////////////////////////
    // Xout = Matrix_ToolBox::Matrix_Mul(Matrix_ToolBox::Matrix_Transpose(Xin, ControllPnt_Num, 3),
    //                                   Xout, dims);
    // Xin  = Matrix_ToolBox::Matrix_Mul(Matrix_ToolBox::Matrix_Transpose(Xin, ControllPnt_Num, 3),
    //                                   Xin, dims);
    // dims = new int [4] {3, 3, 3, 3};
    // Matrix_ToolBox::Inverse_Cal(Xin, Buff_Inv, 3);
    // Transition_Mx = Matrix_ToolBox::Matrix_Mul(Buff_Inv, Xout, dims);
    // Matrix_ToolBox::Inverse_Cal(Transition_Mx, Transition_Mx_Rev, 3);
    ////////////////////////This Method is deprecated because of in-correctness////////////////////////////
}
vector<unsigned char> ee569_hw3_sol::Fetch_ConditionalMask_BASE(unsigned char BOND_NUM){
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
            ret_Vec = {0b11100011, 0b11111000, 0b00111110, 0b10001111};
            break;
        case 9:     // STK
            ret_Vec = {0b11110011, 0b11100111, 0b11111100, 0b11111001,
                       0b01111110, 0b00111111, 0b10011111, 0b11001111};
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
    vector<unsigned char> ret_Vec, Buff_Vec, BOND_IDX;
    switch(TYPE){
        case SHRINKING:
            BOND_IDX = {1, 2, 3, 42, 51, 52, 61, 62, 7, 8, 9, 10};
            break;
        case THINNING:
            BOND_IDX = {41, 42, 51, 52, 61, 62, 7, 8, 9, 10};
            // BOND_IDX = {42, 52, 62, 7, 8, 9};
            break;
        case SKELETONIZING:
            BOND_IDX = {41, 42, 62, 7, 8, 9, 10, 11};
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
    bool FOR_K = ~ FOR_ST;
    unsigned char * Buff_pnt;
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
    ret_Vec.push_back({0b00111000, 0b00111000, MASK_FOR_NOABC});
    if(FOR_K){
        ret_Vec.push_back({0b10000011, 0b10000011, MASK_FOR_NOABC});
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
    else if(FOR_K){
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
        Buff_Bool1 = ((Byte_Input & M) ^ M01) == 0x00;
        if(MA == MASK_FOR_NOABC){
            Buff_Bool2 = true;
        }
        else{
            Buff_Bool2 = (Byte_Input & MA) != 0x00;
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
void ee569_hw3_sol::Morph_DO(unsigned char ** scr_Img, int height, int width, int maxIter, unsigned char TYPE){
    unsigned char ** Buff_Mask, Buff_Byte;
    vector<unsigned char> CondiMask_Pool = Fetch_ConditionalMask(TYPE);
    vector< array<unsigned char, 3> > UncondiMask_Pool = Fetch_UnconditionalMask(TYPE != 0x02);
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
                if(~Match_UnconditionalMask(UncondiMask_Pool, Buff_Byte))
                    scr_Img[i][j] = 0x00;
            }
        }
        delete[] Buff_Mask;
    }
}

///////////////////////////////////////////////////NEWLY DEPRECATED CODE///////////////////////////////////////////////////////////////
vector<unsigned char> ee569_hw3_sol::Fetch_Mask(bool IF_CONDITION, unsigned char TYPE){
    vector<unsigned char> ret_Vec, diff_Vec;
    if(TYPE == SHRINKING){
        // Same for Conditional & Unconditional
        ret_Vec =  {0b00000001,0b00000100,0b01000000,0b00010000,                                             // Bond1
                    0b10000000,0b00100000,0b00001000,0b00000010,                                             // Bond2
                    0b11000000,0b01100000,0b00110000,0b00011000,0b00001100,0b00000110,0b00000011,0b10000001, // Bond3
                    0b11000001,0b01110000,0b00011100,0b00000111,                                             // Bond4
                    0b11100000,0b00001110,                                                                   // Bond5
                    0b11110000,0b11100001,0b01111000,0b00111100,0b00011110,0b00001111,0b10000111,0b11000011, // Bond6
                    0b11110001,0b01111100,0b00011111,0b11000111};                                            // Bond7
        // Different
        if(IF_CONDITION){
            diff_Vec = {0b10000011,                                                                          // Bond5
                        0b10001111, 0b11100011,                                                              // Bond8
                        0b11110011, 0b11100111, 0b10011111, 0b11001111};                                     // Bond9
        }
        else if(~ IF_CONDITION){
            diff_Vec = {0b00111000,                                                                          // Bond5
                        0b00111110, 0b11111000,                                                              // Bond8
                        0b00111111, 0b11111100, 0b11111001, 0b01111110};                                     // Bond9
        }
    }
    else if(TYPE == THINNING){
        ret_Vec =  {0b11000001,0b01110000,0b00011100,0b00000111,                                             // Bond4
                    0b11100000,0b00001110,                                                                   // Bond5
                    0b11110000,0b11100001,0b01111000,0b00111100,0b00011110,0b00001111,0b10000111,0b11000011, // Bond6
                    0b11110001,0b01111100,0b00011111,0b11000111};                                            // Bond7
        if(IF_CONDITION){
            diff_Vec = {0b10000011,                                                                          // Bond5
                        0b10001111, 0b11100011,                                                              // Bond8
                        0b11110011, 0b11100111, 0b10011111, 0b11001111};                                     // Bond9
        }
        else if(~ IF_CONDITION){
            diff_Vec = {0b00111000,                                                                          // Bond5
                        0b00111110, 0b11111000,                                                              // Bond8
                        0b00111111, 0b11111100, 0b11111001, 0b01111110};                                     // Bond9
        }
    }
    else if(TYPE == SKELETONIZING){
        if(IF_CONDITION){
            ret_Vec =  {0b11110000,0b11100001,0b01111000,0b00111100,0b00011110,0b00001111,0b10000111,0b11000011,  // Bond6
                        0b11110001,0b01111100,0b00011111,0b11000111,                                              // Bond7
                        0b11100011,0b11111000,0b00111110,0b10001111,                                              // Bond8
                        0b11110011,0b11100111,0b11111100,0b11111001,0b01111110,0b00111111,0b10011111,0b11001111}; // Bond9
                        // 0b11110111,0b11111101,0b01111111,0b11011111,
                        // 0b10111111,0b11101111,0b11111011,0b11111110};
        }
        else if(~ IF_CONDITION){
            ret_Vec =  {//0b10101111, 0b11101011, 0b11111010, 0b01111110};
                        0b00000001,0b00000100,0b01000000,0b00010000,
                        0b11000001, 0b01110000, 0b00011100, 0b00000111};
        }

    }
    ret_Vec.insert(ret_Vec.end(), diff_Vec.begin(), diff_Vec.end());
    return ret_Vec;
}

bool ee569_hw3_sol::Match_Mask(vector<unsigned char> Mask_Pool, unsigned char Byte_Input){
    for(int i = 0; i < Mask_Pool.size(); ++i){
        if((Byte_Input ^ Mask_Pool[i]) == 0x00)
            return true;
    }
    return false;
}

unsigned char ee569_hw3_sol::Neighborhood_Reorder(unsigned char ** scr_Img, int i, int j){
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

unsigned char ee569_hw3_sol::UnconditionalNeighborhood_Cal(unsigned char Byte_X, unsigned char Byte_E){
    return Byte_X & (~ Byte_E);
}

void ee569_hw3_sol::Morph_DO(unsigned char ** scr_Img, int height, int width, int maxIter, unsigned char TYPE){
    unsigned char ** Buff_Mask, Buff_Byte, ** Buff_Result;
    vector<unsigned char> CondiMask_Pool = Fetch_Mask(true, TYPE);
    vector<unsigned char> UncondiMask_Pool = Fetch_Mask(false, TYPE);
    for(int iter = 0; iter < maxIter; ++iter){
        Buff_Mask = AllocateImg_Uchar(height + 4, width + 4, 1)[0];
        Buff_Result = AllocateImg_Uchar(height + 4, width + 4, 1)[0];
        for(int i = 0; i < height + 4; ++i){
            for(int j = 0; j < width + 4; ++j){
                Buff_Result[i][j] = scr_Img[i][j];
            }
        }
        // Conditional Masking
        if(TYPE == THINNING || TYPE == SHRINKING){
            for(int i = 1; i < height + 3; ++i){
                for(int j = 1; j < width + 3; ++j){
                    if(scr_Img[i][j] == 0x00)
                        continue;
                    Buff_Byte = Neighborhood_Reorder(scr_Img, i, j);
                    Buff_Mask[i][j] = 255 * (unsigned char) Match_Mask(CondiMask_Pool, Buff_Byte);
                }
            }
            for(int i = 2; i < height + 2; ++i){
                for(int j = 2; j < width + 2; ++j){
                    if(scr_Img[i][j] == 0x00 || Buff_Mask[i][j] == 0x00)
                        continue;
                    Buff_Byte = Neighborhood_Reorder(scr_Img, i, j);
                    Buff_Byte = UnconditionalNeighborhood_Cal(Buff_Byte, Neighborhood_Reorder(Buff_Mask, i, j));
                    if(~ Match_Mask(UncondiMask_Pool, Buff_Byte))
                        Buff_Result[i][j] = 0x00;
                }
            }
        }
        else if(TYPE == SKELETONIZING){
            for(int i = 2; i < height + 2; ++i){
                for(int j = 2; j < width + 2; ++j){
                    if(scr_Img[i][j] == 0x00)
                        continue;
                    Buff_Byte = Neighborhood_Reorder(scr_Img, i, j);
                    if(Match_Mask(CondiMask_Pool, Buff_Byte))
                        Buff_Result[i][j] = 0x00;
                }
            }
        }
        for(int i = 0; i < height + 4; ++i){
            for(int j = 0; j < width + 4; ++j){
                scr_Img[i][j] = Buff_Result[i][j];
            }
        }
        delete[] Buff_Mask; delete[] Buff_Result;
    }
}
