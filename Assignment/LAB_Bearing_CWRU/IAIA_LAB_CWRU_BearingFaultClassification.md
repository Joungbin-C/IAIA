# LAB: Bearing Fault Classification

Industrial AI & Automation 2023



**Name:**  

**Date:** 



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

Write a short abstract for this lab

<img src="https://user-images.githubusercontent.com/38373000/228200357-9c5b14ef-ec7a-4309-981b-4b0f37e1dfd8.png" alt="image" style="zoom:50%;" />

## Data Preparation

Explain how datasets are prepared for training and testing etc



## Preprocessing and Feature Extraction

For Feature Extraction, use

* Statistical Features
* Complex Envelope Analysis
* Wavelet Package Analysis
* Complete Pool: Augmentation of all above features



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

* Features: Use Statistical,  Envelope,   WPT, and  Complete Pool 
* Classifer:  SVM / K-NN / Decision Tree

<img src="https://user-images.githubusercontent.com/38373000/228201484-742f9cc6-a45b-44f0-a776-a0abbd13618e.png" alt="image" style="zoom:70%;" />



### Case 2. Classification  with feature selection/reduction

* Features: Use only Global Pool 
* Selection:  PCA / Forward selection
* Classifer: SVM / K-NN / Decision Tree

<img src="https://user-images.githubusercontent.com/38373000/228202257-9a797efb-5f86-4bff-bdf4-f4ca45f27470.png" alt="image" style="zoom:60%;" />



Then, write a short analysis of the results.

You may add graphics, plots and other evaluation index for analysis, if necessary.



