# Assignment 1:  Pre-Processing & Feature Extraction



## Introduction

This assignment is implementing  'Feature Extraction' Section in the following journal paper: [Download here](https://github.com/ykkimhgu/HGU_IAIA/tree/main/Docs)

Rauber, T. W., de Assis Boldt, F., & Varejao, F. M. (2015, January). Heterogeneous Feature Models and Feature Selection Applied to Bearing Fault Diagnosis. IEEE Transactions on Industrial Electronics. Institute of Electrical and Electronics Engineers (IEEE).


### 1. Read the paper and understand the whole process
This Assignment is implementation a part of Feature Extration in  the literature 
Download here
Rauber, T. W., de Assis Boldt, F., & Varejao, F. M. (2015, January). Heterogeneous Feature Models and Feature Selection Applied to Bearing Fault Diagnosis. IEEE Transactions on Industrial Electronics. Institute of Electrical and Electronics Engineers (IEEE).

### 2. Extract and Analyze  Features
### 3. Compare  Features among different classes


### Other reference
Diagnostics 101: A Tutorial for Fault Diagnostics of Rolling Element Bearing Using Envelope Analysis in MATLAB 

[]: 
[Matlab code]: https://www.kau-sdol.com/matlab-code

With the given dataset, 

1. Extract and Analyze  Features and

2. Compare  Features among different bearing fault classes

   

## Preparation

Download Template MATLAB Source

* IAIA_Assignment_CWRU_FeatureExtraction_student.mlx  ([download here)](https://github.com/ykkimhgu/HGU_IAIA/tree/main/Assignment/Assignment_FeatureExtraction_CWRU)



Download small CWRU Dataset

* CWRU_small_rawData_FeatureExtraction_Assignment.zip
  (**[download here](https://github.com/ykkimhgu/HGU_IAIA/tree/main/Dataset)**)



## Dataset
**Sampling:** 
Drive end bearing(12K): 12,000 samples/second

**Classes:**
Normal / Outer Race fault / Inner Race fault
Each under 1HP load,  fault diameter of 0.007inches

**Variables**
DE - drive end accelerometer data
FE - fan end accelerometer data
BA - base accelerometer data
time - time series data
RPM - shaft rpm during testing






## Extract  and Analyze Features   

*  Extract Time-Domain Features
*  Mean, Std, Skewness, Kurtosis, Peak2Peak, RMS, CrestFactor,  ShapeFactor, ImpulseFactor, MarginFactor, Energy etc
*  Extract Frequency-Domain Features
  *   Basic features,  SKMean, SKStd, SKSkewness, SKKurtosis, etc

*  Envelop Extraction
*  Plot  STFT 
*  Plot  Kurtogram
*  Plot Spectral Kurtosis



## Analyze and Compare Features 





## Report

Submit as a zip file. 

* Assignment_FeatureExtraction_YourName.pdf
  * It should show all the outputs
* Assignment_FeatureExtraction_YourName.mlx
* /Library



>  Do NOT submit the dataset





