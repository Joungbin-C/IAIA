# LAB: Bearing Fault Classification

Industrial AI & Automation 2023



**Name:**  Joungbin Choi	

**Date:** 2025/12/01



---



# Introduction

This lab is implementing a part of the following journal paper 

* Thomas W. Rauber et al. "Heterogeneous Feature Models and Feature Selection Applied to Bearing Fault Diagnosis", IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 62, NO. 1, JANUARY 2015
* [Download the journal pdf file](https://github.com/ykkimhgu/digitaltwinNautomation-src/blob/main/Heterogeneous%20Feature%20Models%20and%20Feature%20Selection%20Applied%20to%20Bearing%20Fault%20Diagnosis.pdf)



## Dataset

For the dataset, we will use  

* the selected CWRU dataset  used in journal
* Download CWRU dataset for this lab: [Download here](https://drive.google.com/file/d/1pv-0E8hA77Nr5-gHwVgPq3PR2rdyCj_-/view?usp=sharing)

<img src="https://user-images.githubusercontent.com/38373000/160838885-b74dc1af-4bc9-4bd1-a76f-0bff7f5dd00a.png" alt="image-20220328150553353" style="zoom:80%;" />

## Report

The file name should be  **LAB_BearingDiagnosisCWRU_YourName.***   // for main src and report



Submit  the report either

(Option 1)   *.mlx & mat files  (Recommended) 

(Option 2)  or     *.md and *.mat files.  When writing a report in *.md format, you have to embed the code snippets as done in *.mlx file.



Also, submit   **LAB_BearingDiagnosisCWRU_YourName.pdf**  file.





---



# Procedure



## Overview

In this lab, vibration signals from the CWRU bearing dataset were used to compare several feature extraction and classification methods for fault diagnosis. Statistical features, envelope analysis, and wavelet packet features were extracted and combined into a global feature set. Feature reduction was performed using PCA and forward selection, and the reduced features were classified using KNN, SVM, and Decision Tree. The results show how different feature types and selection methods affect classification accuracy and help identify bearing fault conditions more effectively.

<img src="https://user-images.githubusercontent.com/38373000/228200357-9c5b14ef-ec7a-4309-981b-4b0f37e1dfd8.png" alt="image" style="zoom:50%;" />

## Data Preparation

The dataset consists of vibration signals measured at the Drive-end and Fan-end sensors. There are multiple operating conditions exist:

- normal, ball fault, inner race fault, outer race fault
- different loads(0~3 HP)
- different fault diameters(0.007, 0.014, 0.021 in)

Each experimented data is split to around 2400 samples(50 segments)

`cvpartition(Y, 'KFold',10)` is used to use 9 folds for train and 1 folds for test.



## Preprocessing and Feature Extraction

For Feature Extraction, use

* Statistical Features(26) : 10 features from time domain + 3 features from frequency domain
* Complex Envelope Analysis(72) : 
  * High-pass filter at 500 Hz
  * Hilbert transform
  * FFT of Envelope
  * Extract RMS energy in narrowband around BPFI, BPFO, BSF
  *  DE and FE cross detection
* Wavelet Package Analysis(32) : 
  * Mother wavelet : db4
  * Tree depth : 4
  * Leaves : 16 nodes
* Complete Pool: Augmentation of all above features : total 130



## Feature Reduction

For Feature Reduction/Selection, use 

- Sequential **Forward** Selection
- PCA



## Classification

For Classification Use 

* SVM

* KNN

* Decision Tree

  

## Result and Analysis

Classification output should be described with 

* Confusion Matrix  
* Accuracy table



### Case 1.  Classification without feature selection

* Accuracy Table

![class4_acc_table](C:\Users\joung\source\repos\IAIA\Lab\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\class4_acc_table.png)

- Confusion Matrix of Statistical Features

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\statistica_knn.jpg" alt="statistica_knn" style="zoom: 25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\statistica_svm.jpg" alt="statistica_svm" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\statistica_dtree.jpg" alt="statistica_dtree" style="zoom:25%;" />

- Confusion Matrix of Envelope Features

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\envelope_knn.jpg" alt="statistica_knn" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\envelope_svm.jpg" alt="statistica_svm" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\envelope_dtree.jpg" alt="statistica_dtree" style="zoom:25%;" />

- Confusion Matrix of Wavelet

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\wavelet_knn.jpg" alt="wavelet_knn" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\wavelet_svm.jpg" alt="wavelet_svm" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\wavelet_dtree.jpg" alt="wavelet_dtree" style="zoom:25%;" />

- Confusion Matrix of Complete Pool

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\complete_pool_knn.jpg" alt="complete_pool_knn" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\complete_pool_svm.jpg" alt="complete_pool_svm" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 1\complete_pool_dtree.jpg" alt="complete_pool_dtree" style="zoom:25%;" />

### Case 2. Classification  with feature selection/reduction

* Accuracy Table

![feature_acc_table](..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\feature_acc_table.png)

- Confusion Matrix of feature Reduction

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\pca_knn_conf.jpg" alt="pca_knn_conf" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\pca_svm_conf.jpg" alt="pca_svm_conf" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\pca_dtree_conf.jpg" alt="pca_dtree_conf" style="zoom:25%;" />

- Confusion Matrix of Feature Selection(SFS)

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\sfs_knn_conf.jpg" alt="sfs_knn_conf" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\sfs_svm_conf.jpg" alt="sfs_svm_conf" style="zoom:25%;" />

<img src="..\Lab_BearingDiagnosisCWRU_Joungbin\img\case 2\sfs_dtree_conf.jpg" alt="sfs_dtree_conf" style="zoom:25%;" />



### Analysis

Even though the full 130-feature set already achieves very high classification accuracy, reducing the dimensionality or the number of features still provides meaningful benefits. Techniques such as PCA and SFS can remove redundant or irrelevant features while maintaining nearly the same accuracy, demonstrating that only a subset of core information is sufficient to distinguish between fault conditions. At the same time, feature reduction offers three major advantages: first, it significantly lowers computational cost, improving training and inference speed and making real-time deployment more feasible; second, it reduces overfitting and sensitivity to noise, thereby enhancing the model’s generalization capability; and third, it simplifies system implementation and maintenance by enabling strong performance with a more compact feature set. Thus, feature reduction is valuable not only for accuracy preservation but also for improving efficiency, robustness, and practical usability.



