// Self Defined Header files
# include "ee569_hw4.hpp"
# include "MyUtils.hpp"
# include "Matrix_ToolBox.hpp"
// Standard header files
# include <stdio.h>
# include <iostream>
# include <fstream>
# include <stdlib.h>
# include <math.h>
# include <algorithm>
# include <ctime>
# include <cstdlib>
# include <float.h>
// Public Source Header Files
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"
# include <Eigen/Dense>

using namespace std;

/*
    Stick to standard data structures
*/
int randomSeed(int i){
    return rand() % i;
}
// Model Methods for PCA
myModules::PCA::PCA(vector<vector<float>> DataMx, int Output_Dim){
    int X_height = DataMx.size(), X_width = DataMx[0].size();
    Eigen::MatrixXf X(X_height, X_width);
    for(int i = 0; i < X_height; ++i){
        for(int j = 0; j < X_width; ++j){
            X(i, j) = DataMx[i][j];
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeFullV);
    if(showSingularValues){
        cout << "Its singular values are:" << endl << svd.singularValues() << endl;    
    }
    vector<float> Buff_eigenVec;
    for(int i = 0; i < X_width; ++i){
        Buff_eigenVec.clear();
        for(int j = 0; j < Output_Dim; ++j){
            Buff_eigenVec.push_back(svd.matrixV()(i, j));
        }
        eigenMx.push_back(Buff_eigenVec);
    }
}
void myModules::PCA::fit(vector<vector<float>> DataMx, int Output_Dim){
    int X_height = DataMx.size(), X_width = DataMx[0].size();
    Eigen::MatrixXf X(X_height, X_width);
    for(int i = 0; i < X_height; ++i){
        for(int j = 0; j < X_width; ++j){
            X(i, j) = DataMx[i][j];
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeFullV);
    if(showSingularValues){
        cout << "Its singular values are:" << endl << svd.singularValues() << endl;    
    }
    vector<float> Buff_eigenVec;
    for(int i = 0; i < X_width; ++i){
        Buff_eigenVec.clear();
        for(int j = 0; j < Output_Dim; ++j){
            Buff_eigenVec.push_back(svd.matrixV()(i, j));
        }
        eigenMx.push_back(Buff_eigenVec);
    }
}
std::vector<std::vector<float>> myModules::PCA::predict(std::vector<std::vector<float>> DataMx){
    if(eigenMx.size() == 0)
        throw "Please First Input Dataset";
    return Matrix_ToolBox::Matrix_Mul(DataMx, eigenMx);
}
/*
    Model Methods for Kmeans
*/
void myModules::KMeans::fit(vector<vector<float>> DataMx, vector<int> label){
    /*
        Like Sklearn, we are using restart to get the best clustering
        Criterion used: Inertia / With-in Cluster Variance
    */
    double Inertia = DBL_MAX, Buff_Inertia;
    vector<int> Best_ClusterAssignment;
    srand(unsigned(time(0)));
    for(int trials = 0; trials < restart; ++trials){
        /*
            Initialize the Cluster Assignment
        */
        ClusterAssign_Init(DataMx);
        /*
            Start Iterating
        */
        vector<int> Next_ClusterAssignment = ClusterAssignment;
        int Iter_Counter = 0;
        do{
            ClusterAssignment = Next_ClusterAssignment;
            /*
                Expectation Step, Update Cluster Centers
            */
            ClusterCenter_Update(DataMx);
            /*
                Maximization Step, Update Cluster Assignments
            */
            Next_ClusterAssignment = ClusterAssign_Update(DataMx);
            /*
                Update Counter
            */
            Iter_Counter += 1;
        }while((Next_ClusterAssignment != ClusterAssignment) && (Iter_Counter <= maxIter));
        /*
            Determine Inertia, and compare
        */
        Buff_Inertia = 0.;
        for(int i = 0; i < DataMx.size(); ++i){
            Buff_Inertia += Fetch_Distance(DataMx[i], ClusterCenters[ClusterAssignment[i]]);
        }
        if(Buff_Inertia < Inertia){
            Inertia = Buff_Inertia;
            Best_ClusterAssignment = ClusterAssignment;
        }
    }
    /*
        Update ClusterAssignment & ClusterCenter
    */
    ClusterAssignment = Best_ClusterAssignment;
    ClusterCenter_Update(DataMx);
    /*
        Conduct Majority Vote to the Clusters
        Update ClusterLabels
    */
    vector<vector<int>> Buff_K_TO_CLASS = Matrix_ToolBox::Allocate_Mx_Vec_Int(K, 1 + * max_element(label.begin(), label.end()));
    for(int i = 0; i < DataMx.size(); ++i){
        Buff_K_TO_CLASS[ClusterAssignment[i]][label[i]] += 1;
    }
    for(int i = 0; i < K; ++i){
        ClusterLabels.push_back(max_element(Buff_K_TO_CLASS[i].begin(), Buff_K_TO_CLASS[i].end()) - Buff_K_TO_CLASS[i].begin());
    }
    /*
        Result presentation
    */
    cout << "This is the Distribution of TrainSet & Clusters : " << endl;
    for(int i = 0; i < Buff_K_TO_CLASS.size(); ++i){
        for(int j = 0; j < Buff_K_TO_CLASS[i].size(); ++j){
            cout << Buff_K_TO_CLASS[i][j] << " ";
        }
        cout << "|" << ClusterLabels[i] << "|" << endl;
    }
}
vector<int> myModules::KMeans::predict_Label(vector<vector<float>> DataMx){
    vector<int> ret_Prediction;
    double Buff_Distance;
    for(int i = 0; i < DataMx.size(); ++i){
        Buff_Distance = DBL_MAX;
        ret_Prediction.push_back(K + 10);
        for(int j = 0; j < ClusterCenters.size(); ++j){
            if(Buff_Distance >= Fetch_Distance(DataMx[i], ClusterCenters[j])){
                Buff_Distance = Fetch_Distance(DataMx[i], ClusterCenters[j]);
                ret_Prediction.back() = ClusterLabels[j];
            }
        }
    }
    return ret_Prediction;
}
vector<int> myModules::KMeans::predict_Cluster(vector<vector<float>> DataMx){
    vector<int> ret_Prediction;
    double Buff_Distance;
    for(int i = 0; i < DataMx.size(); ++i){
        Buff_Distance = DBL_MAX;
        ret_Prediction.push_back(K + 10);
        for(int j = 0; j < ClusterCenters.size(); ++j){
            if(Buff_Distance >= Fetch_Distance(DataMx[i], ClusterCenters[j])){
                Buff_Distance = Fetch_Distance(DataMx[i], ClusterCenters[j]);
                ret_Prediction.back() = j;
            }
        }
    }
    return ret_Prediction;
}
double myModules::KMeans::Fetch_Distance(vector<float> Vec1, vector<float> Vec2){
    if(Vec1.size() != Vec2.size())
        throw "The dimensions don't match";
    double ret_Distance = 0.;
    for(int i = 0; i < Vec1.size(); ++i)
        ret_Distance += pow(Vec1[i] - Vec2[i], 2);
    return ret_Distance;
}
vector<int> myModules::KMeans::ClusterAssign_Update(vector<vector<float>> DataMx){
    vector<int> ret_ClusterAssign_Update = Matrix_ToolBox::Allocate_Vec_Vec_Int(DataMx.size());
    double Buff_dis;
    for(int i = 0; i < DataMx.size(); ++i){
        Buff_dis = DBL_MAX;
        for(int j = 0; j < ClusterCenters.size(); ++j){
            if(Buff_dis >= Fetch_Distance(DataMx[i], ClusterCenters[j])){
                Buff_dis = Fetch_Distance(DataMx[i], ClusterCenters[j]);
                ret_ClusterAssign_Update[i] = j;
            }
        }
    }
    return ret_ClusterAssign_Update;
}
void myModules::KMeans::ClusterCenter_Update(vector<vector<float>> DataMx){
    vector<int> Buff_InstanceCounter = Matrix_ToolBox::Allocate_Vec_Vec_Int(K);
    ClusterCenters = Matrix_ToolBox::Allocate_Mx_Vec(K, DataMx[0].size());
    for(int i = 0; i < DataMx.size(); ++i){
        for(int j = 0; j < DataMx[0].size(); ++j){
            ClusterCenters[ClusterAssignment[i]][j] += DataMx[i][j];
        }
        Buff_InstanceCounter[ClusterAssignment[i]] += 1;
    }
    for(int i = 0; i < K; ++i){
        for(int j = 0; j < ClusterCenters.size(); ++j){
            ClusterCenters[i][j] /= Buff_InstanceCounter[i];
        }
    }
}
void myModules::KMeans::ClusterAssign_Init(vector<vector<float>> DataMx){
    /*
        We only need ClusterAssignment Updated, Updating ClusterCenters will be at .fit
    */
    if(Use_Kmeanspp){
        ClusterAssignment.clear();
        ClusterCenters.clear();
        for(int i = 0; i < DataMx.size(); ++i){
            ClusterAssignment.push_back(9);
        }
        // Determine the first center
        int Buff_instance = rand() % (DataMx.size());
        ClusterCenters.push_back(DataMx[Buff_instance]);
        vector<double> Buff_Distance;
        double Buff_dis;
        // Iteratively Initialize the ClustersCenters
        while(ClusterCenters.size() < K){
            Buff_Distance.clear();
            for(int i = 0; i < DataMx.size(); ++i){
                Buff_dis = DBL_MAX;
                for(int j = 0; j < ClusterCenters.size(); ++j){
                    Buff_dis = min(Buff_dis, Fetch_Distance(DataMx[i], ClusterCenters[j]));
                }
                Buff_Distance.push_back(Buff_dis);
            }
            /*
                Partial Sum
                Random Select Pivots, we are utilizing Buff_dis again
            */
            for(int i = 1; i < Buff_Distance.size(); ++i){
                Buff_Distance[i] += Buff_Distance[i - 1];
            }
            Buff_dis = Buff_Distance.back() * (double) rand() / RAND_MAX;
            for(Buff_instance = 0; Buff_instance < Buff_Distance.size(); ++Buff_instance){
                if(Buff_dis <= Buff_Distance[Buff_instance]) break;
            }
            /*
                Push The Next Cluster to ClusterCenters
            */
           ClusterCenters.push_back(DataMx[Buff_instance]);
        }
        /*
            Update ClusterAssignment
            We're reusing Buff_dis
        */
        ClusterAssignment = ClusterAssign_Update(DataMx);
    }
    else if(!Use_Kmeanspp){
        /*
            Assign Cluster Randomly
            We should later address the problem when Dataset size is not divisable by K
        */
        int InstancePerCluster = DataMx.size() / K;
        for(int i = 0; i < K; ++i){
            for(int j = 0; j < InstancePerCluster; ++j){
                ClusterAssignment.push_back(i);
            }
        }

        random_shuffle(ClusterAssignment.begin(), ClusterAssignment.end(), randomSeed);
    }
}
void myModules::KMeans::show_para(){
    cout << "Parameter Settings :" << endl;
    cout << Use_Kmeanspp << " " << K << endl;
    cout << "Cluster Assignment :" << endl;
    for(int i = 0; i < ClusterAssignment.size(); ++i){
        cout << ClusterAssignment[i] << endl;
    }
    cout << "Cluster Center + Cluster Label:" << endl;
    for(int i = 0; i < ClusterCenters.size(); ++i){
        for(int j = 0; j < ClusterCenters[i].size(); ++j){
            cout << ClusterCenters[i][j] << " ";
        }
        cout << ClusterLabels[i] << endl;
    }
}
/*
    Model Methods for Support Vector Machines
*/
myModules::SVM::SVM(){
    svmModel = cv::ml::SVM::create();
    svmModel -> setType(cv::ml::SVM::C_SVC);
    svmModel -> setKernel(cv::ml::SVM::RBF);
    svmModel -> setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
}
void myModules::SVM::fit(vector<vector<float>> DataMx, vector<int> label){
    cv::Mat DataMx_Mat(DataMx.size(), DataMx[0].size(), CV_32FC1);
    cv::Mat label_Mat(label.size(), 1, CV_32S);
    for(int i = 0; i < DataMx.size(); i++){
        for(int j = 0; j < DataMx[i].size(); j++){
            DataMx_Mat.at<float>(i, j) = DataMx[i][j];
        }
        label_Mat.at<int>(i) = label[i];
    }
    svmModel -> trainAuto(cv::ml::TrainData::create(DataMx_Mat, cv::ml::ROW_SAMPLE, label_Mat));
}
vector<int> myModules::SVM::predict(vector<vector<float>> DataMx){
    cv::Mat Buff_Instance(1, DataMx[0].size(), CV_32FC1);
    vector<int> ret_Vec;
    for(int i = 0; i < DataMx.size(); ++i){
        for(int j = 0; j < DataMx[i].size(); ++j){
            Buff_Instance.at<float>(j) = DataMx[i][j];
        }
        ret_Vec.push_back(svmModel -> predict(Buff_Instance));
    }
    return ret_Vec;
}
/*
    Model Methods for Random Forest (Hyperparameters Copied from )
*/
myModules::RandomForest::RandomForest(){
    rfModel = cv::ml::RTrees::create();
    rfModel->setMaxCategories(2);
    rfModel->setMaxDepth(10);
    rfModel->setMinSampleCount(1);
    rfModel->setTruncatePrunedTree(false);
    // rfModel -> setMaxNumOfTreesInTheForest(500);
    rfModel->setUse1SERule(false);
    rfModel->setUseSurrogates(false);
}
void myModules::RandomForest::fit(vector<vector<float>> DataMx, vector<int> label){
    cv::Mat DataMx_Mat(DataMx.size(), DataMx[0].size(), CV_32FC1);
    cv::Mat label_Mat(label.size(), 1, CV_32S);
    for(int i = 0; i < DataMx.size(); i++){
        for(int j = 0; j < DataMx[i].size(); j++){
            DataMx_Mat.at<float>(i, j) = DataMx[i][j];
        }
        label_Mat.at<int>(i) = label[i];
    }
    rfModel -> train(cv::ml::TrainData::create(DataMx_Mat, cv::ml::ROW_SAMPLE, label_Mat));
}
vector<int> myModules::RandomForest::predict(vector<vector<float>> DataMx){
    cv::Mat Buff_Instance(1, DataMx[0].size(), CV_32FC1);
    vector<int> ret_Vec;
    for(int i = 0; i < DataMx.size(); ++i){
        for(int j = 0; j < DataMx[i].size(); ++j){
            Buff_Instance.at<float>(j) = DataMx[i][j];
        }
        ret_Vec.push_back(rfModel -> predict(Buff_Instance));
    }
    return ret_Vec;
}

// Fetch Kernels for Feature Extraction
vector<myKernel> ee569_hw4_sol::Fetch_Kernel(){
    vector<myKernel> ret_Vec;
    float Kernel_1DPool[5][5] = {{1., 4., 6., 4., 1.},
                                 {-1., -2., 0., 2., 1.},
                                 {-1., 0., 2., 0., -1.},
                                 {-1., 2., 0., -2., 1.},
                                 {1., -4., 6., -4., 1.}};
    int dims[2] = {5, 5};
    for(int i = 0; i < 5; ++i){
        for(int j = 0; j < 5; ++j){
            ret_Vec.push_back(myKernel(Matrix_ToolBox::Tensor_Product(Kernel_1DPool[i], Kernel_1DPool[j], dims), 5, 5));
        }
    }
    return ret_Vec;
}
/*
    Fetch Kernels for Segmentation (Averaging), odd sized Kernels
    Since we are using zero padding
    Tips: Uniform Kernels & Gaussian Kernels can all be constructed using Tensor Product / Outer Product
*/
vector<myKernel> ee569_hw4_sol::Fetch_Kernel(unsigned char TYPE, int WindowSize){
    vector<myKernel> ret_Vec;
    float * Kernel_1DPool = new float [WindowSize];
    if(TYPE == GAUSSIAN){
        int center = WindowSize / 2;
        float std = 1., Sum_Weight;
        for(int i = 0; i < WindowSize; ++i){
            Kernel_1DPool[i] = exp(pow(i - center, 2) / (2 * std));
            Sum_Weight += Kernel_1DPool[i];
        }
        for(int i = 0; i < WindowSize; ++i){
            Kernel_1DPool[i] /= Sum_Weight;
        }
    }
    else if(TYPE == UNIFORM){
        for(int i = 0; i < WindowSize; ++i){
            Kernel_1DPool[i] = 1. / WindowSize;
        }
    }
    int dims[2];
    dims[0] = WindowSize; dims[1] = WindowSize;
    ret_Vec.push_back(myKernel(Matrix_ToolBox::Tensor_Product(Kernel_1DPool, Kernel_1DPool, dims), WindowSize, WindowSize));
    return ret_Vec;
}

// Pivot Alignment rule: Upper left
float ee569_hw4_sol::LinearCONV_Op(unsigned char ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j){
    float ret_Result = 0.;
    for(int i = 0; i < Ker.height; ++i){
        for(int j = 0; j < Ker.width; ++j){
            ret_Result += scr_Img[abs(Upperleft_i + i)][abs(Upperleft_j + j)] * Ker(i, j);
        }
    }
    return ret_Result;
}
float ee569_hw4_sol::LinearCONV_Op(float ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j){
    float ret_Result = 0.;
    for(int i = 0; i < Ker.height; ++i){
        for(int j = 0; j < Ker.width; ++j){
            ret_Result += scr_Img[abs(Upperleft_i + i)][abs(Upperleft_j + j)] * Ker(i, j);
        }
    }
    return ret_Result;
}

vector<unsigned char **> ee569_hw4_sol::Read_Data(unsigned char TYPE){
    // First we read all the images
    vector<unsigned char **> dataImg;
    vector<int>              label, trainDim = {128, 128, 1};
    vector<string> filenameList = {"blanket", "brick", "grass", "rice"};
    if(TYPE == ONLY_TRAIN || TYPE == ALL_DATA){
        for(int i = 0; i < filenameList.size(); ++i){
            for(int j = 1; j <= 9; ++j){
                dataImg.push_back(myUtils::input_raw("../scrRaw/" + filenameList[i] + to_string(j),
                                                    trainDim[0], trainDim[1], trainDim[2], 2, "", true)[0]);
                label.push_back(i);
            }
        }
    }
    if(TYPE == ONLY_TEST || TYPE == ALL_DATA){
        for(int i = 1; i <= 12; ++i){
            dataImg.push_back(myUtils::input_raw("../scrRaw/" + to_string(i), trainDim[0], trainDim[1], trainDim[2], 2, "", true)[0]);
            label.push_back(9);
        }
    }
    // We will find a way to attach the labels;
    if(TYPE == SEG_DATA){
        dataImg.push_back(myUtils::input_raw("../scrRaw/comp", 450, 600, 1, 2, "", true)[0]);
    }
    return dataImg;
}
vector<int> ee569_hw4_sol::Read_Label(unsigned char TYPE){
    vector<int> ret_Label;
    vector<string> filenameList = {"blanket", "brick", "grass", "rice"};
    if(TYPE == ONLY_TRAIN || TYPE == ONLY_TEST || TYPE == ALL_DATA){
        for(int i = 0; i < filenameList.size(); ++i){
            for(int j = 1; j <= 9; ++j){
                ret_Label.push_back(i);
            }
        }
    }
    if(TYPE == SEG_DATA){
        for(int i = 0; i < 450 * 600; ++i){
            ret_Label.push_back(0);
        }
    }
    return ret_Label;
}
vector<float ***> ee569_hw4_sol::Feature_Extraction(vector<unsigned char **> dataset,
                                                    vector<myKernel> Pool_Kernel,
                                                    vector<int> dim){

    vector<float ***> ret_FeatureVec;
    float *** Buff_FeatureVec;
    int Img_height = dim[0], Img_width = dim[1], featureVec_Size = Pool_Kernel.size();
    // We only need to extract mean for the first kernel
    vector<double> Img_Mean;
    for(int Img_Pnt = 0; Img_Pnt < dataset.size(); ++Img_Pnt){
        Img_Mean.push_back(0.);
        for(int i = 0; i < Img_height; ++i){
            for(int j = 0; j < Img_width; ++j){
                Img_Mean.back() += (double) dataset[Img_Pnt][i][j]; 
            }
        }
        Img_Mean.back() /= (Img_height * Img_width);
    }
    // Feature Extraction
    for(int Img_Pnt = 0; Img_Pnt < dataset.size(); ++Img_Pnt){
        Buff_FeatureVec = myUtils::AllocateImg_Float(Img_height, Img_width, featureVec_Size);
        for(int Ker_Pnt = 0; Ker_Pnt < featureVec_Size; ++Ker_Pnt){
            for(int i = 0; i < Img_height; ++i){
                for(int j = 0; j < Img_width; ++j){
                    Buff_FeatureVec[Ker_Pnt][i][j] = LinearCONV_Op(dataset[Img_Pnt], Pool_Kernel[Ker_Pnt], i, j);
                    if(Ker_Pnt == 0)        Buff_FeatureVec[Ker_Pnt][i][j] -= float(Img_Mean[Img_Pnt] * 256);
                }
            }
        }
        ret_FeatureVec.push_back(Buff_FeatureVec);
    }
    return ret_FeatureVec;
}
vector<vector<float>> ee569_hw4_sol::Feature_Average(vector<float ***> Pool_FeatureVec,
                                                     vector<myKernel> Pool_Kernel,
                                                     vector<int> dim, bool FOR_CLASSIFICATION){
    double Buff_Avg;
    int Img_height = dim[0], Img_width = dim[1], Img_Channel = dim[2], KernelNum_1D = sqrt(Pool_Kernel.size());
    /*
        We first do 25D -> 15D
    */
    vector<float ***>   Pool_FeatureVec_15D;
    float ***           Buff_Pool_FeatureVec_15D;
    int                 Pnt_FeatureVec_15D;
    for(int Img_Pnt = 0; Img_Pnt < Pool_FeatureVec.size(); ++Img_Pnt){
        Buff_Pool_FeatureVec_15D = myUtils::AllocateImg_Float(Img_height, Img_width, 15);
        for(int i = 0; i < Img_height; ++i){
            for(int j = 0; j < Img_width; ++j){
                Pnt_FeatureVec_15D = 0;
                for(int Ker_Pnti = 0; Ker_Pnti < KernelNum_1D; ++Ker_Pnti){
                    for(int Ker_Pntj = Ker_Pnti; Ker_Pntj < KernelNum_1D; ++Ker_Pntj){
                        Buff_Avg  = abs(Pool_FeatureVec[Img_Pnt][KernelNum_1D * Ker_Pnti + Ker_Pntj][i][j]);
                        Buff_Avg += abs(Pool_FeatureVec[Img_Pnt][Ker_Pnti + KernelNum_1D * Ker_Pntj][i][j]);
                        Buff_Pool_FeatureVec_15D[Pnt_FeatureVec_15D][i][j] = Buff_Avg / 2;
                        Pnt_FeatureVec_15D += 1;
                    }
                }
            }
        }
        Pool_FeatureVec_15D.push_back(Buff_Pool_FeatureVec_15D);
    }
    /*
        We then average over different Dimensions, for different purposes
    */
    vector<vector<float>> ret_AvgFeatureVec;
    vector<float> Buff_AvgFeatureVec;
    if(FOR_CLASSIFICATION){
        for(int Img_Pnt = 0; Img_Pnt < Pool_FeatureVec_15D.size(); ++Img_Pnt){
            Buff_AvgFeatureVec.clear();
            for(int Feature_Pnt = 0; Feature_Pnt < 15; ++Feature_Pnt){
                Buff_Avg = 0.;
                for(int i = 0; i < Img_height; ++i){
                    for(int j = 0; j < Img_width; ++j){
                        Buff_Avg += abs(Pool_FeatureVec_15D[Img_Pnt][Feature_Pnt][i][j]);
                    }
                }
                Buff_AvgFeatureVec.push_back(float(Buff_Avg / Img_height / Img_width));
            }
            ret_AvgFeatureVec.push_back(Buff_AvgFeatureVec);
        }
    }
    else{
        int padding = 1;
        int dim_Pad[4] = {1, 1, 1, 1};
        vector<myKernel>    Kernel_Avg = Fetch_Kernel(UNIFORM, 2 * padding + 1);
        for(int Img_Pnt = 0; Img_Pnt < Pool_FeatureVec_15D.size(); ++Img_Pnt){
            // Do Boundary Extention With Zero-Padding
            Pool_FeatureVec_15D[Img_Pnt] = myUtils::Image_Pad(Pool_FeatureVec_15D[Img_Pnt], Img_height, Img_width, 15, dim_Pad);
            // For each pixel
            for(int i = 0; i < Img_height; ++i){
                for(int j = 0; j < Img_width; ++j){
                    Buff_AvgFeatureVec.clear();
                    // For each Channel
                    for(int k = 0; k < 15; ++k){
                        Buff_Avg = LinearCONV_Op(Pool_FeatureVec_15D[Img_Pnt][k], Kernel_Avg[0], i, j);
                        Buff_AvgFeatureVec.push_back( (float) Buff_Avg);
                    }
                    ret_AvgFeatureVec.push_back(Buff_AvgFeatureVec);
                }
            }
        }
    }
    return ret_AvgFeatureVec;
}
vector<vector<float>> ee569_hw4_sol::Feature_Preprocess(unsigned char Scope_Dataset, bool FOR_CLASSIFICATION){
    /*
        First We Define Some Global Dataset Information
    */
    vector<unsigned char **> Pool_Img = Read_Data(Scope_Dataset);
    vector<int>              Dim_Img;
    if(FOR_CLASSIFICATION)        Dim_Img = {128, 128, 1};
    else if(! FOR_CLASSIFICATION) Dim_Img = {450, 600, 1};
    vector<myKernel>         Pool_Kernel = Fetch_Kernel();
    int Img_height = Dim_Img[0], Img_width = Dim_Img[1], Img_channel = Dim_Img[2];
    int Size_Dataset = Pool_Img.size(), Size_FeatureVec = Pool_Kernel.size();
    /*
        Mean Extraction + Feature Extraction
        Pool_FeatureVec: Img_Num * Kernel_Num * Height * Width
    */
    vector<float ***> Pool_FeatureVec = Feature_Extraction(Pool_Img, Pool_Kernel, Dim_Img);
    /*
        Feature Averaging
        Avg_FeatureVec: Img_Num * Feature_Num (For Classification) / Pixel_Num * Feature_Num (For Segmentation)
    */
    vector<vector<float>> Pool_AvgFeatureVec = Feature_Average(Pool_FeatureVec, Pool_Kernel, Dim_Img, FOR_CLASSIFICATION);
    Size_Dataset = Pool_AvgFeatureVec.size();
    Size_FeatureVec = Pool_AvgFeatureVec[0].size();
    /*
        This is the section for significance analysis
        Divide every row with it's first element (L5L5)
    */
    for(int i = 0; i < Size_Dataset; ++i){
       for(int j = 1; j < Size_FeatureVec; ++j){
           Pool_AvgFeatureVec[i][j] /= Pool_AvgFeatureVec[i][0];
       }
    }
    return Pool_AvgFeatureVec;
}
void ee569_hw4_sol::Feature_Visualization(){
    /*
        This Function serves to output result for Q1(a)
        Feature Reduction PCA & Visualization
        Pool_PCAFeatureVec: Img_Num * outputDim
    */   
    vector<vector<float>> Pool_Feature = Feature_Preprocess(ONLY_TRAIN, true);
    vector<vector<float>> Pool_Test = Feature_Preprocess(ONLY_TEST, true);

    int outputDim = 3;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    
    Pool_Feature = pcaModel.predict(Pool_Feature);
    Pool_Test = pcaModel.predict(Pool_Test);
        
    ofstream csvFile("../output.csv");
    for(int i = 0; i < outputDim; ++i){
        for(int j = 0; j < Pool_Feature.size(); ++j){
            csvFile << (float) Pool_Feature[j][i] << ", ";
        }
        csvFile << endl;
    }
    csvFile.close();
}
/*
    Label encoding chart:
    blanket = 0; brick = 1; grass = 2; rice = 3;
*/
void ee569_hw4_sol::TextureClassification_Unsupervised(){
    // KMEANS_UNIT_TEST();
    /*
        Test Dataset Human Labeling (True Label):
            2 0 0 1 3 2 1 3 3 1 0 2
        Prediction Trials:
            3 0 0 1 3 3 0 3 3 0 1 3 From (C++) (acc = 6 / 12)
            0 0 0 1 3 3 0 3 3 0 1 3 From (Python Sklearn) (acc = 6 / 12)
        Main Question:
            Too much 3, too much 2 -> 3 ======> Come up with tricks in .pred step
    */
    vector<vector<float>> Pool_Feature = Feature_Preprocess(ONLY_TRAIN, true);
    vector<vector<float>> Pool_Test = Feature_Preprocess(ONLY_TEST, true);
    vector<int>           Pool_Label = Read_Label(ONLY_TRAIN);
    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    Pool_Test = pcaModel.predict(Pool_Test);
    /*
        Output Dataset for Python Scripts
    */
    // cout << "This is the Train Dataset" << endl;
    // for(int i = 0; i < Pool_Feature.size(); ++i){
    //     cout << "[" << Pool_Feature[i][0];
    //     for(int j = 1; j < Pool_Feature[i].size(); ++j){
    //         cout << "," << Pool_Feature[i][j];
    //     }
    //     cout << "]," << endl;
    // }
    // cout << "This is the Test Dataset" << endl;
    // for(int i = 0; i < Pool_Test.size(); ++i){
    //     cout << "[" << Pool_Test[i][0];
    //     for(int j = 1; j < Pool_Test[i].size(); ++j){
    //         cout << "," << Pool_Test[i][j];
    //     }
    //     cout << "]," << endl;
    // }
    /*
        Predict Labels
    */
    myModules::KMeans kmeans(true, 4, 200);
    kmeans.fit(Pool_Feature, Pool_Label);
    vector<int> y_pred = kmeans.predict_Label(Pool_Test);
    cout << "This is the Prediction from Kmeans Clustering :" << endl;
    for(int i = 0; i < y_pred.size(); ++i){
        cout << y_pred[i] << " ";   
    }
    cout << endl;
}
void ee569_hw4_sol::TextureClassification_Supervised_SVM(){
    /*
        Test Dataset Human Labeling (True Label):
            2 0 0 1 3 2 1 3 3 1 0 2
        Prediction Trials:
            2 0 0 1 3 2 0 3 3 1 1 2  From (C++) (acc = 10 / 12)
             From (Python Sklearn) (acc = 6 / 12)
        Main Question:
    */
    vector<vector<float>> Pool_Feature = Feature_Preprocess(ONLY_TRAIN, true);
    vector<vector<float>> Pool_Test = Feature_Preprocess(ONLY_TEST, true);
    vector<int>           Pool_Label = Read_Label(ONLY_TRAIN);
    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    Pool_Test = pcaModel.predict(Pool_Test);
    /*
        Predict Labels
    */
    myModules::SVM svm;
    svm.fit(Pool_Feature, Pool_Label);
    vector<int> y_pred = svm.predict(Pool_Test);
    cout << "This is the Prediction from Support Vector Machines :" << endl;
    for(int i = 0; i < y_pred.size(); ++i){
        cout << y_pred[i] << " ";   
    }
    cout << endl;
}
void ee569_hw4_sol::TextureClassification_Supervised_RF(){
    /*
        Test Dataset Human Labeling (True Label):
            2 0 0 1 3 2 1 3 3 1 0 2
        Prediction Trials:
            2 0 0 1 3 2 1 3 3 1 1 2  From (C++) (acc = 11 / 12)
             From (Python Sklearn) (acc = 6 / 12)
        Main Question:
    */
    vector<vector<float>> Pool_Feature = Feature_Preprocess(ONLY_TRAIN, true);
    vector<vector<float>> Pool_Test = Feature_Preprocess(ONLY_TEST, true);
    vector<int>           Pool_Label = Read_Label(ONLY_TRAIN);
    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    Pool_Test = pcaModel.predict(Pool_Test);
    /*
        Predict Labels
    */
    myModules::RandomForest rf;
    rf.fit(Pool_Feature, Pool_Label);
    vector<int> y_pred = rf.predict(Pool_Test);
    cout << "This is the Prediction from Random Forest :" << endl;
    for(int i = 0; i < y_pred.size(); ++i){
        cout << y_pred[i] << " ";   
    }
    cout << endl;
}
/*
    Texture Segmentation
*/
void ee569_hw4_sol::TextureClassification_Segmentation(){
    vector<vector<float>>    Pool_Feature = Feature_Preprocess(SEG_DATA, false);
    vector<int>              Pool_Label = Read_Label(SEG_DATA);
    vector<int>              Dim_Img = {450, 600, 1};
    /*
        PCA
    */    
    int outputDim = 3;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    /*
        Predict Labels
    */

    // myModules::KMeans kmeans(true, 6, 200);
    // kmeans.fit(Pool_Feature, Pool_Label);
    // vector<int> y_pred = kmeans.predict_Cluster(Pool_Feature);


    unsigned char *** ret_Seg = myUtils::AllocateImg_Uchar(Dim_Img[0], Dim_Img[1], Dim_Img[2]);
    for(int k = 0; k < Dim_Img[2]; ++k){
        for(int i = 0; i < Dim_Img[0]; ++i){
            for(int j = 0; j < Dim_Img[1]; ++j){
                ret_Seg[k][i][j] = 51 * (unsigned char) y_pred[0];
                y_pred.erase(y_pred.begin());
            }
        }
    }
    myUtils::output_pgm("../compPred", ret_Seg, Dim_Img[0], Dim_Img[1], Dim_Img[2]);
}



/*
    Utility Functions & Unit Tests
*/
void ee569_hw4_sol::READ_ALL_RAW(){
    /*
        Section for 36 + 12 Images in Q1(a)
    */
    vector<string> filenameList = {"blanket", "brick", "grass", "rice"};
    vector<array<int, 3>> dimList;
    unsigned char *** Buff_Img;
    for(int i = 0; i < 4; ++i){
        for(int j = 1; j <= 9; ++j){
            filenameList.push_back(filenameList[i] + to_string(j));
            dimList.push_back({128, 128, 1});
        }
    }
    for(int i = 1; i <= 12; ++i){
        filenameList.push_back(to_string(i));
        dimList.push_back({128, 128, 1});
    }
    filenameList.erase(filenameList.begin(), filenameList.begin() + 4);
    /*
        Section for the composite Image in Q1(b)
    */
   filenameList.push_back("comp");
   dimList.push_back({450, 600, 1});
    // Converting
    for(int i = 0; i < filenameList.size(); ++i){
        Buff_Img = myUtils::input_raw("../scrRaw/" + filenameList[i], dimList[i][0], dimList[i][1], dimList[i][2], 0, "", true);
        if(dimList[i][2] == 1)
            myUtils::output_pgm("../scrPxm/" + filenameList[i], Buff_Img, dimList[i][0], dimList[i][1], dimList[i][2]);
        else if(dimList[i][2] == 3)
            myUtils::output_ppm("../scrPxm/" + filenameList[i], Buff_Img, dimList[i][0], dimList[i][1], dimList[i][2]);
    }

    cout << "All RAW image input succeeded & finished" << endl;
}

void ee569_hw4_sol::KMEANS_UNIT_TEST(){
    /*
        Test Log: Use-Kmeans++ == false (2 labels) passed;
                  Use-Kmeans++ == true  (2 labels) passed;
                  Use-Kmeans++ == false (3 labels) passed;
                  Use-Kmeans++ == true  (3 labels) passed;
    */
    vector<vector<float>> X = {{10, 0}, {9.5, 0.5}, {-10, 9},
                               {-10, -9}, {-15, 8}, {-8, -6},
                               {-9, 8}, {-9, -10}, {11, 2}};
    vector<int>           y = {0, 0, 2, 1, 2, 1, 2, 1, 0};
    myModules::KMeans kmeans(true, 3, 100);
    kmeans.fit(X, y);
    vector<int>           y_pred = kmeans.predict_Label(X);
    for(int i = 0; i < y_pred.size(); ++i){
        cout << y_pred[i] << endl;
    }
    kmeans.show_para();
}
