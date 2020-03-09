#ifndef EE569_HW4_HPP
#define EE569_HW4_HPP


# include <stdio.h>
# include <iostream>
# include "opencv2/core.hpp"
# include "opencv2/ml.hpp"


# include "MyUtils.hpp"
# include "Matrix_ToolBox.hpp"

# define ONLY_TRAIN 0x00
# define ONLY_TEST  0X01
# define ALL_DATA   0x02
# define SEG_DATA   0x03
# define GAUSSIAN   0xF0
# define UNIFORM    0xF1


int randomSeed(int i);
cv::Mat MAT_TO_VEC(std::vector<std::vector<float>> Vec_Mx);
std::vector<std::vector<float>> VEC_TO_MAT(cv::Mat Mat_Mx);

class ee569_hw4_sol{
    public:
        ee569_hw4_sol(){}
        ~ee569_hw4_sol(){}

        void READ_ALL_RAW();

        void Feature_Visualization();
        void TextureClassification_Unsupervised();
        void TextureClassification_Supervised_SVM();
        void TextureClassification_Supervised_RF();
        void TextureClassification_Segmentation();
        
    private:
        std::vector<myKernel> Fetch_Kernel();
        std::vector<myKernel> Fetch_Kernel(unsigned char TYPE, int WindowSize);
        float LinearCONV_Op(unsigned char ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j);
        float LinearCONV_Op(float ** scr_Img, myKernel Ker, int Upperleft_i, int Upperleft_j);
        std::vector<std::vector<float>> PCA(std::vector<std::vector<float>> DataMx, int Output_Dim);
        std::vector<unsigned char **> Read_Data(unsigned char TYPE);
        std::vector<int>              Read_Label(unsigned char TYPE);
        std::vector<float ***>          Feature_Extraction(std::vector<unsigned char **> dataset, 
                                                           std::vector<myKernel> Pool_Kernel,
                                                           std::vector<int> dim);
        std::vector<std::vector<float>> Feature_Average(std::vector<float ***> Pool_FeatureVec,
                                                        std::vector<myKernel> Pool_Kernel,
                                                        std::vector<int> dim, bool FOR_CLASSIFICATION);
        std::vector<std::vector<float>> Feature_Preprocess(unsigned char Scope_Dataset, bool FOR_CLASSIFICATION);
        void KMEANS_UNIT_TEST();
};

namespace myModules{
    class PCA{
        public:
            PCA(){}
            ~PCA(){}
            bool showSingularValues = false;
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
            void show_para();
            
        private:
            bool Use_Kmeanspp; int K; int restart; int maxIter = 200;
            std::vector<std::vector<float>> ClusterCenters;
            std::vector<int>                ClusterAssignment;
            std::vector<int>                ClusterLabels;
            void ClusterAssign_Init(std::vector<std::vector<float>> DataMx);
            double Fetch_Distance(std::vector<float> Vec1, std::vector<float> Vec2);
            std::vector<int> ClusterAssign_Update(std::vector<std::vector<float>> DataMx);
            void             ClusterCenter_Update(std::vector<std::vector<float>> DataMx);
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
}


#endif
