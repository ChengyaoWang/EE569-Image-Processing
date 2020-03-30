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
# include <exception>
// Public Source Header Files
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"
# include "opencv2/highgui.hpp"
# include "opencv2/features2d.hpp"
# include "opencv2/xfeatures2d.hpp"
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
        We only want to Initialize the Cluster Assignment
        Initialize All the Class Variables
    */
    ClusterAssignment = Matrix_ToolBox::Allocate_Vec_Vec_Int(DataMx.size());
    ClusterAssignment_Next = Matrix_ToolBox::Allocate_Vec_Vec_Int(DataMx.size());
    ClusterLabels = Matrix_ToolBox::Allocate_Vec_Vec_Int(K);
    InstanceCounter = Matrix_ToolBox::Allocate_Vec_Vec_Int(K);
    KMeanspp_Prob = Matrix_ToolBox::Allocate_Vec_Vec_2F(DataMx.size());
    /*
        Like Sklearn, we are using restart to get the best clustering
        Criterion used: Inertia / With-in Cluster Variance
    */
    double Inertia = DBL_MAX, Buff_Inertia;
    vector<int> Best_ClusterAssignment;
    srand(unsigned(time(0)));
    for(int trials = 0; trials < restart; ++trials){
        cout << "The " << trials + 1 << "th Number of Trial: " << endl;
        /*
            Initialize the Cluster Assignment
        */
        ClusterAssign_Init(DataMx);
        /*
            Start Iterating
        */
        int Iter_Counter = 0;
        ClusterAssignment_Next = ClusterAssignment;
        do{
            ClusterAssignment = ClusterAssignment_Next;
            /*
                Expectation Step, Update Cluster Centers
            */
            ClusterCenter_Update(DataMx);
            /*
                Maximization Step, Update Cluster Assignments
            */
            ClusterAssign_Update(DataMx);
            /*
                Update Counter
            */
            Iter_Counter += 1;
        }while((ClusterAssignment_Next != ClusterAssignment) && (Iter_Counter <= maxIter));
        cout << "Completed with " << Iter_Counter << " / " << maxIter << " Number of Iterations" << endl;
        /*
            Determine Inertia, and compare
        */
        Buff_Inertia = 0.;
        for(int i = 0; i < DataMx.size(); ++i){
            Buff_Inertia += Fetch_Distance(DataMx[i], ClusterCenters[ClusterAssignment[i]]);
        }
        cout << "With Terminating Inertia = " << Buff_Inertia << endl;
        cout << "-----------------------------------------------" << endl;
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
        ClusterLabels[i] = max_element(Buff_K_TO_CLASS[i].begin(), Buff_K_TO_CLASS[i].end()) - Buff_K_TO_CLASS[i].begin();
    }
    /*
        Result presentation
    */
    if(Print_Label_Cluster_Distribution){
        cout << "This is the Distribution of TrainSet & Clusters : " << endl;
        for(int i = 0; i < Buff_K_TO_CLASS.size(); ++i){
            for(int j = 0; j < Buff_K_TO_CLASS[i].size(); ++j){
                cout << Buff_K_TO_CLASS[i][j] << " ";
            }
            cout << "|" << ClusterLabels[i] << "|" << endl;
        }
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
void myModules::KMeans::ClusterAssign_Update(const vector<vector<float>> & DataMx){
    Matrix_ToolBox::Clear_Matrix(ClusterAssignment_Next, 0);
    double Buff_dis;
    for(int i = 0; i < DataMx.size(); ++i){
        Buff_dis = DBL_MAX;
        for(int j = 0; j < ClusterCenters.size(); ++j){
            if(Buff_dis >= Fetch_Distance(DataMx[i], ClusterCenters[j])){
                Buff_dis = Fetch_Distance(DataMx[i], ClusterCenters[j]);
                ClusterAssignment_Next[i] = j;
            }
        }
    }
}
void myModules::KMeans::ClusterCenter_Update(const vector<vector<float>> & DataMx){
    Matrix_ToolBox::Clear_Matrix(InstanceCounter, 0);
    Matrix_ToolBox::Clear_Matrix(ClusterCenters, 0.);
    for(int i = 0; i < DataMx.size(); ++i){
        for(int j = 0; j < DataMx[0].size(); ++j){
            ClusterCenters[ClusterAssignment[i]][j] += DataMx[i][j];
        }
        InstanceCounter[ClusterAssignment[i]] += 1;
    }
    for(int i = 0; i < ClusterCenters.size(); ++i){
        for(int j = 0; j < ClusterCenters[0].size(); ++j){
            ClusterCenters[i][j] /= InstanceCounter[i];
        }
    }
}
void myModules::KMeans::ClusterAssign_Init(const vector<vector<float>> & DataMx){
    Matrix_ToolBox::Clear_Matrix(ClusterAssignment, 0);
    Matrix_ToolBox::Clear_Matrix(ClusterAssignment_Next, 0);
    Matrix_ToolBox::Clear_Matrix(ClusterLabels, 0);
    Matrix_ToolBox::Clear_Matrix(InstanceCounter, 0);
    Matrix_ToolBox::Clear_Matrix(KMeanspp_Prob, 0.);
    ClusterCenters.clear();
    /*
        Use Kmeans++ or not
    */
    if(Use_Kmeanspp){
        // Determine the first center
        int Buff_instance = rand() % (DataMx.size());
        ClusterCenters.push_back(DataMx[Buff_instance]);
        // Iteratively Initialize the ClustersCenters
        double Buff_dis;
        while(ClusterCenters.size() < K){
            Matrix_ToolBox::Clear_Matrix(KMeanspp_Prob, 0.);
            for(int i = 0; i < DataMx.size(); ++i){
                Buff_dis = DBL_MAX;
                for(int j = 0; j < ClusterCenters.size(); ++j){
                    Buff_dis = min(Buff_dis, Fetch_Distance(DataMx[i], ClusterCenters[j]));
                }
                KMeanspp_Prob[i] = Buff_dis;
            }
            /*
                Partial Sum
                Random Select Pivots, we are utilizing Buff_dis again
            */
            for(int i = 1; i < KMeanspp_Prob.size(); ++i){
                KMeanspp_Prob[i] += KMeanspp_Prob[i - 1];
            }
            Buff_dis = KMeanspp_Prob.back() * (double) rand() / RAND_MAX;
            for(Buff_instance = 0; Buff_instance < KMeanspp_Prob.size(); ++Buff_instance){
                if(Buff_dis <= KMeanspp_Prob[Buff_instance]) break;
            }
            /*
                Push The Next Cluster to ClusterCenters
            */
           ClusterCenters.push_back(DataMx[Buff_instance]);
        }
        /*
            Update ClusterAssignment
        */
        ClusterAssign_Update(DataMx);
        ClusterAssignment = ClusterAssignment_Next;
    }
    else if(!Use_Kmeanspp){
        /*
            Assign Cluster Randomly
            We should later address the problem when Dataset size is not divisable by K
        */
        ClusterCenters = Matrix_ToolBox::Allocate_Mx_Vec(K, DataMx[0].size());
        int InstancePerCluster = DataMx.size() / K;
        for(int i = 0; i < K; ++i){
            for(int j = 0; j < InstancePerCluster; ++j){
                ClusterAssignment[i] = j;
            }
        }
        random_shuffle(ClusterAssignment.begin(), ClusterAssignment.end(), randomSeed);
    }
}
void myModules::KMeans::show_para(){
    // cout << "Parameter Settings :" << endl;
    // cout << Use_Kmeanspp << " " << K << endl;
    // cout << "Cluster Assignment :" << endl;
    // for(int i = 0; i < ClusterAssignment.size(); ++i){
    //     cout << ClusterAssignment[i] << endl;
    // }
    // cout << "Cluster Center + Cluster Label:" << endl;
    // for(int i = 0; i < ClusterCenters.size(); ++i){
    //     for(int j = 0; j < ClusterCenters[i].size(); ++j){
    //         cout << ClusterCenters[i][j] << " ";
    //     }
    //     cout << ClusterLabels[i] << endl;
    // }
}
vector<vector<float>> myModules::KMeans::Fetch_ClusterMeans(){
    vector<vector<float>>       ret_Mx = Matrix_ToolBox::Allocate_Mx_Vec(ClusterCenters.size(),
                                                                         ClusterCenters[0].size());
    vector<float>           Buff_Vec;
    for(int i = 0; i < ClusterCenters.size(); ++i){
        Buff_Vec.clear();
        for(int j = 0; j < ClusterCenters[i].size(); ++j){
            Buff_Vec.push_back(ClusterCenters[i][j]);
        }
        ret_Mx.push_back(Buff_Vec);
    }
    return ret_Mx;
}
vector<int> myModules::KMeans::Fetch_ClusterAssignment(){
    vector<int>     ret_Vec;
    for(int i = 0; i < ClusterAssignment.size(); ++i){
        ret_Vec.push_back(ClusterAssignment[i]);
    }
    return ret_Vec;
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
/*
    Model Methods for Feature Extractor
    Fetch Lowe Kernels for Feature Extraction
    &
    Fetch Kernels for Segmentation (Averaging), odd sized Kernels
    Uniform Kernels & Gaussian Kernels can all be constructed using Tensor Product / Outer Product
*/
vector<myKernel> myModules::FeatureExtracter::Fetch_Kernel(){
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
vector<myKernel> myModules::FeatureExtracter::Fetch_Kernel(unsigned char TYPE, int WindowSize){
    vector<myKernel> ret_Vec;
    float * Kernel_1DPool = new float [WindowSize];
    if(TYPE == MY_GAUSSIAN){
        int center = WindowSize / 2;
        float std = 0.5, Sum_Weight;
        for(int i = 0; i < WindowSize; ++i){
            Kernel_1DPool[i] = exp(pow(i - center, 2) / (2 * std));
            Sum_Weight += Kernel_1DPool[i];
        }
        for(int i = 0; i < WindowSize; ++i){
            Kernel_1DPool[i] /= Sum_Weight;
        }
    }
    else if(TYPE == MY_UNIFORM){
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
double myModules::FeatureExtracter::LinearCONV_Op(unsigned char ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j){
    double ret_Result = 0.;
    for(int i = 0; i < Ker.height; ++i){
        for(int j = 0; j < Ker.width; ++j){
            ret_Result += (double) scr_Img[i + Upperleft_i][j + Upperleft_j] * Ker(i, j);
        }
    }
    return ret_Result;
}
double myModules::FeatureExtracter::LinearCONV_Op(float ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j){
    double ret_Result = 0.;
    for(int i = 0; i < Ker.height; ++i){
        for(int j = 0; j < Ker.width; ++j){
            ret_Result += (double) scr_Img[i + Upperleft_i][j + Upperleft_j] * Ker(i, j);
        }
    }
    return ret_Result;
}
float *** myModules::FeatureExtracter::Feature_Extraction(unsigned char ** scrImg, const vector<myKernel> & Feature_Ker){
    /*
        Extract Features from Input Image Batch
        Input:  (HEIGHT + 2) * (WIDTH + 2)
        Output: 25 * HEIGHT * WIDTH
    */
    float *** Buff_Feature = myUtils::AllocateImg_Float(HEIGHT, WIDTH, Feature_Ker.size());
    for(int i = 0; i < HEIGHT; ++i){
        for(int j = 0; j < WIDTH; ++j){
            for(int k = 0; k < 25; ++k){
                Buff_Feature[k][i][j] = (float) LinearCONV_Op(scrImg, Feature_Ker[k], i, j);
            }
        }
    }
    return Buff_Feature;
}
float *** myModules::FeatureExtracter::Feature_Averaging(float *** Feature25D){
    /*
        Fold 25D -> 15D
    */
    float *** Buff_Feature = myUtils::AllocateImg_Float(HEIGHT, WIDTH, 15);
    int Vec_Pnt = 0;
    for(int Ker_i = 0; Ker_i < 5; ++Ker_i){
        for(int Ker_j = Ker_i; Ker_j < 5; ++Ker_j){
            for(int i = 0; i < HEIGHT; ++i){
                for(int j = 0; j < WIDTH; ++j){
                    Buff_Feature[Vec_Pnt][i][j] += abs(Feature25D[Ker_i + 5 * Ker_j][i][j]);
                    Buff_Feature[Vec_Pnt][i][j] += abs(Feature25D[5 * Ker_i + Ker_j][i][j]);
                    Buff_Feature[Vec_Pnt][i][j] /= 2;
                }
            }
            Vec_Pnt += 1;
        }
    }
    return Buff_Feature;
}
void myModules::FeatureExtracter::Feature_Standardization(vector<vector<float>> & DataMx){
    int InstanceNum = DataMx.size(), FeatureVecNum = DataMx[0].size();
    vector<double>     MeanVec, VarVec;
    if(normalizationType == USE_STANDARD_NORMALIZATION){
        for(int j = 0; j < FeatureVecNum; ++j){
            MeanVec.push_back(0.);
            for(int i = 0; i < InstanceNum; ++i){
                MeanVec.back() += (double) DataMx[i][j];
            }
            MeanVec.back() /= InstanceNum;
        }
        for(int j = 0; j < FeatureVecNum; ++j){
            VarVec.push_back(0.);
            for(int i = 0; i < InstanceNum; ++i){
                VarVec.back() += pow((double) DataMx[i][j] - MeanVec[j], 2);
            }
            VarVec.back() /= (InstanceNum - 1);
        }
        for(int i = 0; i < InstanceNum; ++i){
            for(int j = 0; j < FeatureVecNum; ++j){
                DataMx[i][j] = (DataMx[i][j] - MeanVec[j]) / VarVec[j];
            }
        }
    }
    else if(normalizationType == USE_BASIC_NORMALIZATION){
        for(int i = 0; i < InstanceNum; ++i){
            for(int j = FeatureVecNum - 1; j >= 0; --j){
                DataMx[i][j] /= DataMx[i][0];
            }
        }
    }
    else if(normalizationType == USE_SET_FEAC0_TO_1){
        for(int i = 0; i < InstanceNum; ++i){
            DataMx[i][0] = 1.;
        }
    }
    else if(normalizationType == USE_NOTHING){

    }

}
void myModules::FeatureExtracter::Mean_Subtraction(vector<float ***> & Feature25D){
    if(Use_GlobalMean){
        for(int Img_Ptr = 0; Img_Ptr < Feature25D.size(); ++Img_Ptr){
            for(int i = 0; i < HEIGHT; ++i){
                for(int j = 0; j < WIDTH; ++j){
                    Feature25D[Img_Ptr][0][i][j] -= 256 * (float) GlobalMean[Img_Ptr];
                }
            }
        }
    }
    else if(! Use_GlobalMean){
        for(int Img_Ptr = 0; Img_Ptr < Feature25D.size(); ++Img_Ptr){
            for(int i = 0; i < HEIGHT; ++i){
                for(int j = 0; j < WIDTH; ++j){
                    Feature25D[Img_Ptr][0][i][j] -= 256 * (float) WindowMean_PerImg[Img_Ptr][i][j];
                }
            }
        }
    }
}
vector<vector<float>> myModules::FeatureExtracter::fit(vector<unsigned char **> & ImageBatch){
    /*
        First We Define Some Global Dataset Information, and Fetch the Kernel We're Using
        If we're using Global Mean, Precompute it
        Otherwise if we're using Window Mean, Just push zero
    */
    vector<myKernel>        Feature_Kernel = Fetch_Kernel();
    vector<myKernel>        Average_Kernel = Fetch_Kernel(MY_UNIFORM, WindowSize);
    int Size_Dataset = ImageBatch.size(), Size_FeatureVec = Feature_Kernel.size();
    int Pad_Dims_Feature[4] = {2, 2, 2, 2}, Pad_Dims_Avg[4];
    Pad_Dims_Avg[0] = Padding; Pad_Dims_Avg[1] = Padding; Pad_Dims_Avg[2] = Padding; Pad_Dims_Avg[3] = Padding;
    /*
        Extract GlobalMean OR WindowMean
    */
    if(Use_GlobalMean){
        for(int Img_Ptr = 0; Img_Ptr < ImageBatch.size(); ++Img_Ptr){
            GlobalMean.push_back(0.);
            for(int i = 0; i < HEIGHT; ++i){
                for(int j = 0; j < WIDTH; ++j){
                    GlobalMean.back() += ImageBatch[Img_Ptr][i][j];
                }
            }
            GlobalMean.back() /= (HEIGHT * WIDTH);
        }
    }
    else if(!Use_GlobalMean){
        unsigned char ** Buff_PaddedImg;
        vector<vector<double>> Buff_WindowMean_Img;
        vector<double>         Buff_WindowMean_Row;
        for(int Img_Ptr = 0; Img_Ptr < ImageBatch.size(); ++Img_Ptr){
            Buff_PaddedImg = myUtils::Image_Pad(ImageBatch[Img_Ptr], HEIGHT, WIDTH, Pad_Dims_Avg, false);
            Buff_WindowMean_Img.clear();
            for(int i = 0; i < HEIGHT; ++i){
                Buff_WindowMean_Row.clear();
                for(int j = 0; j < WIDTH; ++j){
                    Buff_WindowMean_Row.push_back(LinearCONV_Op(Buff_PaddedImg, Average_Kernel[0], i, j));
                }
                Buff_WindowMean_Img.push_back(Buff_WindowMean_Row);
            }
            WindowMean_PerImg.push_back(Buff_WindowMean_Img);
        }
    }
    cout << "HandShake: Mean Initialized" << endl;
    // Pad the Image Before hand & Do Feature Extraction
    vector<float ***> Feature25D;
    for(int Img_Ptr = 0; Img_Ptr < ImageBatch.size(); ++Img_Ptr){
        ImageBatch[Img_Ptr] = myUtils::Image_Pad(ImageBatch[Img_Ptr], HEIGHT, WIDTH, Pad_Dims_Feature, false);
        // We load Global Mean, And will modify it if We're using WindowMean
        Feature25D.push_back(Feature_Extraction(ImageBatch[Img_Ptr], Feature_Kernel));
    }
    cout << "HandShake: Feature25D Generated" << endl;
    // Substract Mean
    Mean_Subtraction(Feature25D);
    cout << "HandShake: Mean Subtracted" << endl;
    // 25 -> 15D
    vector<float ***> Feature15D;
    float *** Buff_Feature;
    int Vec_Pnt;
    for(int Img_Ptr = 0; Img_Ptr < ImageBatch.size(); ++Img_Ptr){
        Feature15D.push_back(Feature_Averaging(Feature25D[Img_Ptr]));
    }
    cout << "HandShake: Feature15D Generated" << endl;
    // Generate the Final Dataset
    vector<vector<float>>   Pool_AvgFeatureVec;
    vector<float>           Buff_Pool_AvgFeatureVec;
    for(int Img_Ptr = 0; Img_Ptr < ImageBatch.size(); ++Img_Ptr){
        if(!FOR_CLASSIFICATION){
            Feature15D[Img_Ptr] = myUtils::Image_Pad(Feature15D[Img_Ptr], HEIGHT, WIDTH, 15, Pad_Dims_Avg, false);
            HEIGHT += 2 * Padding;
            WIDTH  += 2 * Padding;
        }
        for(int i = 0; i < HEIGHT + 1 - WindowSize; ++i){
            for(int j = 0; j < WIDTH + 1 - WindowSize; ++j){
                Buff_Pool_AvgFeatureVec.clear();
                for(int k = 0; k < 15; ++k){
                    Buff_Pool_AvgFeatureVec.push_back((float) LinearCONV_Op(Feature15D[Img_Ptr][k], Average_Kernel[0], i, j));
                }
                Pool_AvgFeatureVec.push_back(Buff_Pool_AvgFeatureVec);
            }
        }
    }
    cout << "HandShake: Final Dataset Generated" << endl;
    /*
        This is the section for significance analysis
        Divide every row with it's first element (L5L5)
    */
    Feature_Standardization(Pool_AvgFeatureVec);
    // Zero mean shift
    vector<double>      meanVec;
    for(int j = 0; j < Pool_AvgFeatureVec[0].size(); ++j){
        meanVec.push_back(0.);
        for(int i = 0; i < Pool_AvgFeatureVec.size(); ++i){
            meanVec.back() += (double) Pool_AvgFeatureVec[i][j];
        }
        meanVec.back() /= Pool_AvgFeatureVec.size();
    }
    for(int j = 0; j < Pool_AvgFeatureVec[0].size(); ++j){
        for(int i = 0; i < Pool_AvgFeatureVec.size(); ++i){
            Pool_AvgFeatureVec[i][j] -= meanVec[j];
        }
    }
    return Pool_AvgFeatureVec;
}
void myModules::FeatureExtracter::TENSOR_PRODUCT_UNIT_TEST(){
    vector<myKernel>    Kernel_Pool = Fetch_Kernel();
    for(int i = 0; i < Kernel_Pool.size(); ++i){
        Kernel_Pool[i].show();
    }
    Kernel_Pool = Fetch_Kernel(MY_UNIFORM, 11);
    for(int i = 0; i < Kernel_Pool.size(); ++i){
        Kernel_Pool[i].show();
    }
}
/*
    Method for SIFT
*/
myModules::mySIFT::mySIFT(int nfeatures){
    sift_FeatureExtractor = cv::xfeatures2d::SIFT::create(nfeatures, 3, 0.04, 10, 1.6);
}
inline bool myModules::mySIFT::Compare_Matching(cv::DMatch Match1, cv::DMatch Match2){
    return bool(Match1.distance < Match2.distance);
}
/*
    Read Image & Extract Features / Descriptors
*/
void myModules::mySIFT::fit(string FileName){
    const cv::Mat InputImg = cv::imread("../scrRaw/" + FileName, cv::IMREAD_COLOR);
    Descriptor.clear();
    KeyPoints.clear();
    sift_FeatureExtractor -> detectAndCompute(InputImg, cv::noArray(), KeyPoints, Buff_descriptor);
    Descriptor = Matrix_ToolBox::Allocate_Mx_Vec(Buff_descriptor.rows, Buff_descriptor.cols);
    for(int i = 0; i < Buff_descriptor.rows; ++i){
        for(int j = 0; j < Buff_descriptor.cols; ++j){
            Descriptor[i][j] = Buff_descriptor.at<float> (i, j);
        }
    }
    Buff_descriptor.release();
    cv::drawKeypoints(InputImg, KeyPoints, OutputCanvas, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // cv::drawKeypoints(InputImg, KeyPoints, OutputCanvas, cv::Scalar::all(-1), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../" + FileName, OutputCanvas);
}
void myModules::mySIFT::Print_Keypoints(cv::KeyPoint Input_Keypoint){
    cout << "Angle: " << Input_Keypoint.angle << endl;
    cout << "Octave: " << Input_Keypoint.octave << endl;
    cout << "Coordinate: " << Input_Keypoint.pt.y << "|" << Input_Keypoint.pt.x << endl;
    cout << "Response: " << Input_Keypoint.response << endl;
    cout << "Size: " << Input_Keypoint.size << endl; 
}
void myModules::mySIFT::Print_Matching(string Img1, string Img2, int Min_Match){
    const cv::Mat img1 = cv::imread("../scrRaw/" + Img1, cv::IMREAD_COLOR);
    const cv::Mat img2 = cv::imread("../scrRaw/" + Img2, cv::IMREAD_COLOR);
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift_FeatureExtractor -> detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift_FeatureExtractor -> detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    // Match the Features, Use L2_distance, Filter it
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    vector<vector<cv::DMatch> > knn_matches;
    vector<cv::DMatch>          good_matches;
    // Match the Keypoints, Sort According to Distances
    cv::Mat img_matches;
    matcher -> match(descriptors1, descriptors2, good_matches);
    sort(good_matches.begin(), good_matches.end(), Compare_Matching);
    // Draw Matches
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../" + Img1 + "|" + Img2, img_matches);
}
void myModules::mySIFT::Print_LargestScaleKP_Matching(string Query, string Train){
    // Load Query Image Information to Class Variables
    fit(Query);
    const cv::Mat Query_Img = cv::imread("../scrRaw/" + Query, cv::IMREAD_COLOR);
    const cv::Mat Train_Img = cv::imread("../scrRaw/" + Train, cv::IMREAD_COLOR);
    vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift_FeatureExtractor -> detectAndCompute(Query_Img, cv::noArray(), keypoints1, descriptors1);
    sift_FeatureExtractor -> detectAndCompute(Train_Img, cv::noArray(), keypoints2, descriptors2);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    vector<cv::DMatch>          good_matches;
    matcher -> match(descriptors1, descriptors2, good_matches);
    /*
        Fetch the Largest Scale KeyPoint, And Plot
    */
    float Best_Response = FLT_MIN, Buff_Response;
    int Best_Idx = 0;
    for(int i = 0 ; i < good_matches.size(); ++i){
        Buff_Response = KeyPoints[good_matches[i].queryIdx].response;
        if(Buff_Response > Best_Response){
            Best_Response = Buff_Response;
            Best_Idx = i;
        }
    }
    cout << "Largest Response: " << Best_Response << endl;
    good_matches.push_back(good_matches[Best_Idx]);
    good_matches.erase(good_matches.begin(), good_matches.end() - 1);
    cv::Mat img_matches;
    drawMatches(Query_Img, keypoints1, Train_Img, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("../MaxScale|" + Query + "|" + Train, img_matches);
}
int myModules::mySIFT::Fetch_Descriptor(vector<vector<float>> & Corpus){
    for(int i = 0; i < Descriptor.size(); ++i){
        Corpus.push_back(Descriptor[i]);
    }
    return Descriptor.size();
}

//////////////////////////////////FOR ASSIGNEMENT 4//////////////////////////////////
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
vector<unsigned char **> ee569_hw4_sol::Read_Data(unsigned char TYPE){
    /*
        First we read all the images, padding are done in CONV stage
        We will never turn on zero padding
    */
    vector<unsigned char **> dataImg;
    vector<int>              label, trainDim = {128, 128, 1};
    vector<string> filenameList = {"blanket", "brick", "grass", "rice"};
    if(TYPE == ONLY_TRAIN || TYPE == ALL_DATA){
        for(int i = 0; i < filenameList.size(); ++i){
            for(int j = 1; j <= 9; ++j){
                dataImg.push_back(myUtils::input_raw("../scrRaw/" + filenameList[i] + to_string(j),
                                                    trainDim[0], trainDim[1], trainDim[2], 0, "", false)[0]);
                label.push_back(i);
            }
        }
    }
    if(TYPE == ONLY_TEST || TYPE == ALL_DATA){
        for(int i = 1; i <= 12; ++i){
            dataImg.push_back(myUtils::input_raw("../scrRaw/" + to_string(i),
                              trainDim[0], trainDim[1], trainDim[2], 0, "", false)[0]);
            label.push_back(9);
        }
    }
    // We will find a way to attach the labels;
    if(TYPE == SEG_DATA){
        dataImg.push_back(myUtils::input_raw("../scrRaw/comp", 450, 600, 1, 0, "", false)[0]);
    }
    return dataImg;
}
unsigned char *** ee569_hw4_sol::Segmentation_Reorder(vector<int> & Prediction, int dims[3]){
    unsigned char *** ret_Seg = myUtils::AllocateImg_Uchar(dims[0], dims[1], dims[2]);
    for(int k = 0; k < dims[2]; ++k){
        for(int i = 0; i < dims[0]; ++i){
            for(int j = 0; j < dims[1]; ++j){
                ret_Seg[k][i][j] = 51 * (unsigned char) Prediction[0];
                Prediction.erase(Prediction.begin());
            }
        }
    }
    return ret_Seg;
}
double ee569_hw4_sol::Fetch_Distance(vector<float> Vec1, vector<float> Vec2){
    if(Vec1.size() != Vec2.size())
        throw "The dimensions don't match";
    double ret_Distance = 0.;
    for(int i = 0; i < Vec1.size(); ++i)
        ret_Distance += pow(Vec1[i] - Vec2[i], 2);
    return ret_Distance;
}
void ee569_hw4_sol::Feature_Visualization(){
    /*
        This Function serves to output result for Q1(a)
    */   
    vector<unsigned char **>    X_Train = Read_Data(ONLY_TRAIN);
    vector<unsigned char **>    X_Test  = Read_Data(ONLY_TEST);
    vector<int>                 y_Train = Read_Label(ONLY_TRAIN);
    int dims[3] = {128, 128, 1};
    myModules::FeatureExtracter FeatureGenerator(dims, 128, true, true);
    vector<vector<float>> Feature_Train = FeatureGenerator.fit(X_Train);
    vector<vector<float>> Feature_Test  = FeatureGenerator.fit(X_Test);
    /*
        Discriminant Power Analysis, Features are organized as
        L5L5 - L5E5 - L5S5 - L5W5 - L5R5
        E5E5 - E5S5 - E5W5 - E5R5
        S5S5 - S5W5 - S5R5
        W5W5 - W5R5
        R5R5
    */
    double NAIVE_MEAN = 0.;
    double FISHER_MEAN = 0.;
    // Naive Way
    vector<double>      stdVec;
    for(int j = 0; j < Feature_Train[0].size(); ++j){
        stdVec.push_back(0.);
        for(int i = 0; i < Feature_Train.size(); ++i){
            stdVec.back() += (double) pow(Feature_Train[i][j], 2);
        }
        stdVec.back() /= Feature_Train.size();
    }
    cout << "Unsupervised Feature Variance For Naive Way" << endl;
    for(int i = 0; i < stdVec.size(); ++i){
        NAIVE_MEAN += stdVec[i];
        cout << stdVec[i] << endl;
    }
    // Fisher Linear Discriminant Analysis
    // Sigma: 15 by 15, Sigma_B: 15 by 15, ClassMean: (4 + 1) by 15
    vector<vector<double>>  sigma = Matrix_ToolBox::Matrix_Mul(Feature_Train, Matrix_ToolBox::Matrix_Transpose(Feature_Train));
    vector<vector<double>>  sigma_B = Matrix_ToolBox::Allocate_Mx_Vec_2F(Feature_Train[0].size(), Feature_Train[0].size());
    vector<vector<double>>  classMean = Matrix_ToolBox::Allocate_Mx_Vec_2F(5, Feature_Train[0].size());
    for(int i = 0; i < Feature_Train.size(); ++i){
        for(int j = 0; j < Feature_Train[i].size(); ++j){
            classMean[y_Train[i]][j] += Feature_Train[i][j];
            classMean.back()[j] += Feature_Train[i][j];
        }
    }
    for(int j = 0; j < classMean[0].size(); ++j){
        for(int i = 0; i < classMean.size() - 1; ++i){ 
            classMean[i][j] /= 9.;
        }
        classMean.back()[j] /= 36.;
    }
    // Compute Sigma B:
    for(int CLASS = 0; CLASS < 4; CLASS++){
        for(int i = 0; i < classMean[0].size(); ++i){
            for(int j = 0; j < classMean[0].size(); ++j){
                sigma_B[i][j] += (classMean[CLASS][i] - classMean.back()[i]) * (classMean[CLASS][j] - classMean.back()[j])/4.;
            }
        }
    }
    // Compute Class seperability, and plot
    cout << "Supervised Fisher Linear Discriminant" << endl;
    vector<double>  Class_Sep;
    for(int i = 0; i < classMean[0].size(); ++i){
        Class_Sep.push_back(sigma_B[i][i] / sigma[i][i]);
        FISHER_MEAN += Class_Sep.back();
        cout << Class_Sep.back() << endl;
    }
    // Compute Relative, in percentage
    cout << "Unsupervised Feature Variance For Naive Way: Normalized" << endl;
    for(int i = 0; i < stdVec.size(); ++i){
        cout << stdVec[i] / NAIVE_MEAN << endl;
    }
    cout << "Supervised Fisher Linear Discriminant: Normalized" << endl;
    for(int i = 0; i < Class_Sep.size(); ++i){
        cout << Class_Sep[i] / FISHER_MEAN << endl;
    }

    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Feature_Train, outputDim);
    Feature_Train = pcaModel.predict(Feature_Train);
    Feature_Test = pcaModel.predict(Feature_Test);
    /*
        PCA Output Region
    */
    ofstream csvFile("../output.csv");
    for(int i = 0; i < outputDim; ++i){
        for(int j = 0; j < Feature_Train.size(); ++j){
            csvFile << (float) Feature_Train[j][i] << ", ";
        }
        csvFile << endl;
    }
    csvFile.close();
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
}
/*
    Label encoding chart:
    blanket = 0; brick = 1; grass = 2; rice = 3;
*/
void ee569_hw4_sol::TextureClassification_Unsupervised(){
    /*
        Test Dataset Human Labeling (True Label):
            2 0 0 1 3 2 1 3 3 1 0 2
        Prediction Trials:
            2 0 3 0 3 2 2 3 3 2 0 2   From (C++) (acc = 8 / 12)
    */
    vector<unsigned char **>    X_Train = Read_Data(ONLY_TRAIN);
    vector<unsigned char **>    X_Test  = Read_Data(ONLY_TEST);
    vector<int>                 y_Train = Read_Label(ONLY_TRAIN);
    int dims[3] = {128, 128, 1};
    myModules::FeatureExtracter FeatureGenerator(dims, 128, true, true);
    vector<vector<float>> Feature_Train = FeatureGenerator.fit(X_Train);
    vector<vector<float>> Feature_Test  = FeatureGenerator.fit(X_Test);
    /*
        Do PCA
    */
    // int outputDim = 3;
    // myModules::PCA pcaModel(Feature_Train, outputDim);
    // Feature_Train = pcaModel.predict(Feature_Train);
    // Feature_Test = pcaModel.predict(Feature_Test);

    /*
        Predict Labels
    */
    myModules::KMeans kmeans(true, 4, 200);
    kmeans.fit(Feature_Train, y_Train);
    vector<int> y_pred = kmeans.predict_Label(Feature_Test);
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
            2 0 0 1 3 2 1 3 3 1 0 2    From (C++) (acc = 12 / 12)
    */
    vector<unsigned char **>    X_Train = Read_Data(ONLY_TRAIN);
    vector<unsigned char **>    X_Test  = Read_Data(ONLY_TEST);
    vector<int>                 y_Train = Read_Label(ONLY_TRAIN);
    int dims[3] = {128, 128, 1};
    myModules::FeatureExtracter FeatureGenerator(dims, 128, true, true);
    vector<vector<float>> Feature_Train = FeatureGenerator.fit(X_Train);
    vector<vector<float>> Feature_Test  = FeatureGenerator.fit(X_Test);
    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Feature_Train, outputDim);
    Feature_Train = pcaModel.predict(Feature_Train);
    Feature_Test = pcaModel.predict(Feature_Test);
    /*
        Predict Labels
    */
    myModules::SVM svm;
    svm.fit(Feature_Train, y_Train);
    vector<int> y_pred = svm.predict(Feature_Test);
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
            2 0 0 0 3 2 1 3 3 1 0 2   From (C++) (acc = 11 / 12)
    */
    vector<unsigned char **>    X_Train = Read_Data(ONLY_TRAIN);
    vector<unsigned char **>    X_Test  = Read_Data(ONLY_TEST);
    vector<int>                 y_Train = Read_Label(ONLY_TRAIN);
    int dims[3] = {128, 128, 1};
    myModules::FeatureExtracter FeatureGenerator(dims, 128, true, true);
    vector<vector<float>> Feature_Train = FeatureGenerator.fit(X_Train);
    vector<vector<float>> Feature_Test  = FeatureGenerator.fit(X_Test);
    /*
        Do PCA
    */
    int outputDim = 3;
    myModules::PCA pcaModel(Feature_Train, outputDim);
    Feature_Train = pcaModel.predict(Feature_Train);
    Feature_Test = pcaModel.predict(Feature_Test);
    /*
        Predict Labels
    */
    myModules::RandomForest rf;
    rf.fit(Feature_Train, y_Train);
    vector<int> y_pred = rf.predict(Feature_Test);
    cout << "This is the Prediction from Random Forest :" << endl;
    for(int i = 0; i < y_pred.size(); ++i){
        cout << y_pred[i] << " ";   
    }
    cout << endl;
}
void ee569_hw4_sol::TextureClassification_Segmentation(){
    vector<unsigned char **>    SourceImg   = Read_Data(SEG_DATA);
    vector<int>                 PseudoLabel = Read_Label(SEG_DATA);
    // We are using part of the image for debugging
    // int dims[3] = {100, 100, 1};
    int dims[3] = {450, 600, 1};
    myModules::FeatureExtracter FeatureGenerator(dims, 21, false, true);
    vector<vector<float>> Pool_Feature = FeatureGenerator.fit(SourceImg);
    cout << "Feature Extraction Completed, Dataset Size:" << endl;
    cout << Pool_Feature.size() << " " << Pool_Feature[0].size() << endl;
    /*
        Do PCA
    */
    int outputDim = 15;
    myModules::PCA pcaModel(Pool_Feature, outputDim);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    cout << "PCA Completed, Dataset Size:" << endl;
    cout << Pool_Feature.size() << " " << Pool_Feature[0].size() << endl;
    /*
        Predict Labels
    */
    myModules::KMeans kmeans(true, 6, 20);
    kmeans.fit(Pool_Feature, PseudoLabel);
    vector<int> y_pred = kmeans.predict_Cluster(Pool_Feature);
    unsigned char *** ret_Seg = myUtils::AllocateImg_Uchar(dims[0], dims[1], dims[2]);
    for(int k = 0; k < dims[2]; ++k){
        for(int i = 0; i < dims[0]; ++i){
            for(int j = 0; j < dims[1]; ++j){
                ret_Seg[k][i][j] = 51 * (unsigned char) y_pred[0];
                y_pred.erase(y_pred.begin());
            }
        }
    }
    myUtils::output_pgm("../compPred", ret_Seg, dims[0], dims[1], dims[2]);
}
void ee569_hw4_sol::TextureClassification_Segmentation(int WindowSize, bool Use_GlobalMean, int PCA_OutputDims, unsigned char Normtype){
    vector<unsigned char **>    SourceImg   = Read_Data(SEG_DATA);
    vector<int>                 PseudoLabel = Read_Label(SEG_DATA);
    int dims[3] = {450, 600, 1};
    // Feature Extractor
    myModules::FeatureExtracter FeatureGenerator(dims, WindowSize, false, Use_GlobalMean);
    FeatureGenerator.normalizationType = Normtype;
    vector<vector<float>> Pool_Feature = FeatureGenerator.fit(SourceImg);
    // PCA
    myModules::PCA pcaModel(Pool_Feature, PCA_OutputDims);
    Pool_Feature = pcaModel.predict(Pool_Feature);
    // Kmeans Prediction
    myModules::KMeans kmeans(true, 6, 10);
    kmeans.fit(Pool_Feature, PseudoLabel);
    vector<int> y_pred = kmeans.predict_Cluster(Pool_Feature);
    // Output
    unsigned char *** Segmentation_Map = Segmentation_Reorder(y_pred, dims);
    myUtils::output_pgm("../SegTuneResult/" +
                        to_string(WindowSize) + "|" +
                        to_string(Use_GlobalMean) + "|" +
                        to_string(PCA_OutputDims) + "|" +
                        to_string((int)Normtype),
                        Segmentation_Map, dims[0], dims[1], dims[2]);
}
void ee569_hw4_sol::FeatureDetection_SIFT(){
    myModules::mySIFT       sift;
    /*
        First We Outputs Feature Extracted by SIFT
    */
    sift.fit("Husky_1.jpg");
    sift.fit("Husky_2.jpg");
    sift.fit("Husky_3.jpg");
    sift.fit("Puppy_1.jpg");
    /*
        Next We Outputs Feature Matches Q2(b)2
    */
    sift.Print_Matching("Husky_1.jpg", "Husky_2.jpg", 30);
    sift.Print_Matching("Husky_3.jpg", "Husky_1.jpg", 30);
    sift.Print_Matching("Husky_2.jpg", "Husky_3.jpg", 30);
    sift.Print_Matching("Husky_1.jpg", "Puppy_1.jpg", 30);
    sift.Print_Matching("Husky_2.jpg", "Puppy_1.jpg", 30);
    sift.Print_Matching("Husky_3.jpg", "Puppy_1.jpg", 30);
    sift.Print_Matching("Husky_3.jpg", "Husky_4.jpg", 30);
    /*
        Manually NearestNeighbor Search for Largest Scale Q2(b)1
        Train Image: Husky_1, Query Image: Husky_2
    */
    sift.Print_LargestScaleKP_Matching("Husky_3.jpg", "Husky_1.jpg");
    sift.Print_LargestScaleKP_Matching("Husky_1.jpg", "Husky_3.jpg");
    sift.Print_LargestScaleKP_Matching("Husky_3.jpg", "Husky_2.jpg");
    sift.Print_LargestScaleKP_Matching("Husky_2.jpg", "Husky_3.jpg");
    sift.Print_LargestScaleKP_Matching("Husky_2.jpg", "Husky_1.jpg");
    sift.Print_LargestScaleKP_Matching("Husky_1.jpg", "Husky_2.jpg");
}
void ee569_hw4_sol::FeatureDetection_BoW(){
    myModules::mySIFT       sift(0);
    vector<int>             label_num, label;
    /*
        Start Fetching SIFT Descriptors
    */
    vector<vector<float>>   Feature1;
    sift.fit("Husky_1.jpg");
    label_num.push_back(sift.Fetch_Descriptor(Feature1));
    vector<vector<float>>   Feature2;
    sift.fit("Husky_2.jpg");
    label_num.push_back(sift.Fetch_Descriptor(Feature2));
    vector<vector<float>>   Feature3;
    sift.fit("Husky_3.jpg");
    label_num.push_back(sift.Fetch_Descriptor(Feature3));
    vector<vector<float>>   Feature4;
    sift.fit("Puppy_1.jpg");
    label_num.push_back(sift.Fetch_Descriptor(Feature4));
    // Create Pseudo Labels
    for(int i = 0; i < label_num[2]; ++i){
        label.push_back(0.);
    }
    /*
        Kmeans Clustering
    */
    myModules::KMeans          kmeans(true, 8, 20);
    vector<vector<float>>      Histogram = Matrix_ToolBox::Allocate_Mx_Vec(4, 8);
    kmeans.fit(Feature3, label);
    vector<int>                ClusterAssignment = kmeans.Fetch_ClusterAssignment();
    vector<int>                Pred_1 = kmeans.predict_Cluster(Feature1),
                               Pred_2 = kmeans.predict_Cluster(Feature2),
                               Pred_4 = kmeans.predict_Cluster(Feature4);
    // Load the predictions
    for(int i = 0; i < ClusterAssignment.size(); ++i){
        Histogram[2][ClusterAssignment[i]] += (1. / label_num[label[i]]);
    }
    for(int i = 0; i < Pred_1.size(); ++i){
        Histogram[0][Pred_1[i]] += (1. / label_num[0]);
    }
    for(int i = 0; i < Pred_2.size(); ++i){
        Histogram[1][Pred_2[i]] += (1. / label_num[1]);
    }
    for(int i = 0; i < Pred_4.size(); ++i){
        Histogram[3][Pred_4[i]] += (1. / label_num[3]);
    }
    // Print Histograms
    cout << "Features Detected in Each Img" << endl;
    for(int i = 0; i < 4; ++i){
        cout << label_num[i] << " ";
    }
    cout << endl;
    cout << "(Normalized) Histogram of the Images" << endl;
    for(int i = 0; i < Histogram.size(); ++i){
        for(int j = 0; j < Histogram[i].size(); ++j){
            cout << (Histogram[i][j]) << " ";
        }
        cout << endl;
    }
    // Calculate Distances
    cout << "PairWise Distances" << endl;
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            cout << Fetch_Distance(Histogram[i], Histogram[j]) << " ";
        }
        cout << endl;
    }
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
void ee569_hw4_sol::SVD_UNIT_TEST(){
    vector<vector<float>> TestMatrix = {{1, 3, 6, 6},
                                        {2, 3, 7, 10},
                                        {1, 5, 20, 1},
                                        {10, 2, 5, 10},
                                        {5, 2, 7, 5}};
    myModules::PCA pcaModel(TestMatrix, 3);




}
void ee569_hw4_sol::SEG_TUNE(){
    vector<bool>            GlobalMean = {false, true};
    vector<int>             WindowSize = {7, 21, 41};
    // vector<unsigned char>   NormalType = {USE_BASIC_NORMALIZATION, USE_SET_FEAC0_TO_1, USE_STANDARD_NORMALIZATION};
    vector<unsigned char>   NormalType = {USE_SET_FEAC0_TO_1};
    for(int i = 0; i < GlobalMean.size(); ++i){
        for(int j = 0; j < WindowSize.size(); ++j){
            for(int q = 0; q < NormalType.size(); ++q){
                TextureClassification_Segmentation(WindowSize[j], GlobalMean[i], 3, NormalType[q]);
            }
        }
    }
    // TextureClassification_Segmentation(7, false, 15, USE_BASIC_NORMALIZATION);
    // TextureClassification_Segmentation(21, false, 15, USE_BASIC_NORMALIZATION);
    // TextureClassification_Segmentation(41, false, 15, USE_BASIC_NORMALIZATION);
    cout << "Segmentation Tuning Complete" << endl;
}