

# Assignment:  Feature Classification  of CWRU Dataset



## Introduction

Due in 1 week.



**Read the paper and understand the whole process**

This Assignment is implementation a part of classification in the literature 

- Rauber, T. W., de Assis Boldt, F., & Varejao, F. M. (2015, January). Heterogeneous Feature Models and Feature Selection Applied to Bearing Fault Diagnosis. IEEE Transactions on Industrial Electronics. Institute of Electrical and Electronics Engineers (IEEE).

**Apply KNN and SVM  classification methods with all  features** 



**Apply KNN and SVM  classification methods with Selected features by Forward/Backward eliminiation**

- Read here: https://kr.mathworks.com/help/stats/sequentialfs.html

**Apply KNN and SVM  classification with Reduce feature dimension by PCA or LDA**

- Read here:  https://kr.mathworks.com/help/stats/pca.html?lang=en



You have to show necessary steps and plots/data with proper comments

Which gives the best evaluation performance on Test set? Also try to optimize KNN and SVM



## Preparation

Download Template MATLAB Source file ([download here)](https://github.com/ykkimhgu/HGU_IAIA/tree/main/Assignment/Assignment_Classification_%20CWRUsmall)


## Dataset

Given: Feature Extracted from CWRU bearing dataset 

Use the feature extracted results from Assignment: FeatureExtraction CWRU IAIA_Assignment_CWRU_FeatureExtraction_student.mlx

You are provided with train and test dataset consists of CWRU data features. 

You can refer to the previous Assignment: FeatureExtraction CWRU IAIA_Assignment_CWRU_FeatureExtraction_student.mlx

 

[Download here](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Dataset/CWRU_small_featureData_Classfication_Assignment.zip)

- [CWRU_small_featureData_Classfication_Assignment.zip](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Dataset/CWRU_small_featureData_Classfication_Assignment.zip)

Classes:

-  Normal / Outer Race fault / Inner Race fault / Ball fault



The given data is divided by Train set and Test set 



**Note**

Training. K-fold, cross-validation is performed on Train Dataset only.

Test dataset is used for Evaluation.

