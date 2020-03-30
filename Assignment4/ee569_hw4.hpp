#ifndef EE569_HW4_HPP
#define EE569_HW4_HPP


# include <stdio.h>
# include <iostream>
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"
# include "opencv2/xfeatures2d.hpp"

# include "MyUtils.hpp"
# include "Matrix_ToolBox.hpp"

# define ONLY_TRAIN 0x00
# define ONLY_TEST  0X01
# define ALL_DATA   0x02
# define SEG_DATA   0x03
# define MY_GAUSSIAN   0xF0
# define MY_UNIFORM    0xF1
# define USE_STANDARD_NORMALIZATION     0xA0
# define USE_BASIC_NORMALIZATION        0xA1
# define USE_SET_FEAC0_TO_1             0xA2
# define USE_NOTHING                    0xA3


int randomSeed(int i);

class ee569_hw4_sol{
    public:
        ee569_hw4_sol(){}
        ~ee569_hw4_sol(){}

        void READ_ALL_RAW();
        void TENSOR_PRODUCT_UNIT_TEST();
        void KMEANS_UNIT_TEST();
        void SVD_UNIT_TEST();
        void SEG_TUNE();

        void Feature_Visualization();
        void TextureClassification_Unsupervised();
        void TextureClassification_Supervised_SVM();
        void TextureClassification_Supervised_RF();
        void TextureClassification_Segmentation();
        void FeatureDetection_SIFT();
        void FeatureDetection_BoW();
        
    private:
        std::vector<unsigned char **>   Read_Data(unsigned char TYPE);
        std::vector<int>                Read_Label(unsigned char TYPE);
        double                          Fetch_Distance(std::vector<float> Vec1, std::vector<float> Vec2);
        unsigned char ***               Segmentation_Reorder(std::vector<int> & Prediction, int dims[3]);
        void TextureClassification_Segmentation(int WindowSize,
                                                bool Use_GlobalMean,
                                                int PCA_OutputDims,
                                                unsigned char Normtype);
        
};

namespace myModules{
    class FeatureExtracter{
        public:
            FeatureExtracter(int Dims[3], int Window, bool task, bool MeanType){
                WindowSize = Window;
                Padding = Window / 2;
                FOR_CLASSIFICATION = task;
                HEIGHT = Dims[0];
                WIDTH = Dims[1];
                CHANNEL = Dims[2];
                // We forbid the use of Window mean to Classification Task
                Use_GlobalMean = (MeanType || task);
            }
            ~FeatureExtracter(){}
            unsigned char normalizationType = USE_BASIC_NORMALIZATION;
            std::vector<std::vector<float>> fit(std::vector<unsigned char **> & ImageBatch);
            void TENSOR_PRODUCT_UNIT_TEST();

        private:
            bool FOR_CLASSIFICATION, Use_GlobalMean;
            int HEIGHT, WIDTH, CHANNEL;
            int WindowSize, Padding;
            double Buff_Mean;
            std::vector<double>                              GlobalMean;
            std::vector<std::vector<std::vector<double>>>    WindowMean_PerImg;
            std::vector<float>      Buff_FeatureVec;
            std::vector<myKernel> Fetch_Kernel();
            std::vector<myKernel> Fetch_Kernel(unsigned char TYPE, int WindowSize);
            double LinearCONV_Op(unsigned char ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j);
            double LinearCONV_Op(float ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j);
            float *** Feature_Averaging(float *** Feature25D);
            float *** Feature_Extraction(unsigned char ** scrImg, const std::vector<myKernel> & Feature_Ker);
            void      Feature_Standardization(std::vector<std::vector<float>> & DataMx);
            void Mean_Subtraction(std::vector<float ***> & Feature25D);
    };

    class PCA{
        public:
            PCA(){}
            ~PCA(){}
            bool showSingularValues = true;
            PCA(std::vector<std::vector<float>> DataMx, int Output_Dim);
            void fit(std::vector<std::vector<float>> DataMx, int Output_Dim);
            std::vector<std::vector<float>> predict(std::vector<std::vector<float>> DataMx);

        private:
            std::vector<std::vector<float>> eigenMx;
    };
    class KMeans{
        public:
            KMeans(bool Improved_Init, int Cluster_Num, int Trial_Num){
                Use_Kmeanspp = Improved_Init;
                K = Cluster_Num;
                restart = Trial_Num;
            }
            ~KMeans(){}
            void fit(std::vector<std::vector<float>> DataMx, std::vector<int> label);
            std::vector<int> predict_Label(std::vector<std::vector<float>> DataMx);
            std::vector<int> predict_Cluster(std::vector<std::vector<float>> DataMx);

            std::vector<std::vector<float>>     Fetch_ClusterMeans();
            std::vector<int>                    Fetch_ClusterAssignment();
            void show_para();
            
        private:
            bool Use_Kmeanspp, Print_Label_Cluster_Distribution = true;
            int K, restart, maxIter = 1000;

            // Model Parameters
            std::vector<std::vector<float>> ClusterCenters;
            std::vector<int>                ClusterAssignment;
            std::vector<int>                ClusterAssignment_Next;
            // Buffer Variables, initialized in .fit()
            std::vector<int>                ClusterLabels;
            std::vector<int>                InstanceCounter;
            std::vector<double>             KMeanspp_Prob;

            void ClusterAssign_Init(const std::vector<std::vector<float>> & DataMx);
            double Fetch_Distance(std::vector<float> Vec1, std::vector<float> Vec2);
            void ClusterAssign_Update(const std::vector<std::vector<float>> & DataMx);
            void ClusterCenter_Update(const std::vector<std::vector<float>> & DataMx);
    };
    class SVM{
        public:
            SVM();
            ~SVM(){}
            void fit(std::vector<std::vector<float>> DataMx, std::vector<int> label);
            std::vector<int> predict(std::vector<std::vector<float>> DataMx);
        private:
            cv::Ptr<cv::ml::SVM> svmModel;
    };
    class RandomForest{
        public:
            RandomForest();
            ~RandomForest(){}
            void fit(std::vector<std::vector<float>> DataMx, std::vector<int> label);
            std::vector<int> predict(std::vector<std::vector<float>> DataMx);
        private:
            cv::Ptr<cv::ml::RTrees> rfModel;
    };
    class mySIFT{
        public:
            mySIFT(int nfeatures = 0);
            ~mySIFT(){}

            std::vector<std::vector<float>>     Descriptor;
            std::vector<cv::KeyPoint>           KeyPoints;
            
            void fit(std::string FileName);
            void Print_Keypoints(cv::KeyPoint Input_Keypoint);
            void Print_Matching(std::string Img1, std::string Img2, int Min_Match);
            void Print_LargestScaleKP_Matching(std::string Query, std::string Train);
            int Fetch_Descriptor(std::vector<std::vector<float>> & Corpus);

        private:
            cv::Mat OutputCanvas;
            cv::Mat Buff_descriptor;
            cv::Ptr<cv::xfeatures2d::SIFT> sift_FeatureExtractor;
            static bool Compare_Matching(cv::DMatch Match1, cv::DMatch Match2);


    };
}


#endif
