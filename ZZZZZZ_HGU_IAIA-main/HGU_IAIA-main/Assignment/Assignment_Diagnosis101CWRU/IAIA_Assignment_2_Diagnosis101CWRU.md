# Assignment 2: Diagnostics CWRU Dataset 



## Introduction

This assignment is implementing  'Feature Extraction' Section in the following journal paper

 

Diagnostics 101: A Tutorial for Fault Diagnostics of Rolling Element Bearing Using Envelope Analysis in MATLAB 
[Matlab code]: https://www.kau-sdol.com/matlab-code



Follow Section 5 of the journal paper.  Create your own code to analyze the CWRU dataset  as instructed in the Section 5 of the paper.

Features should include 

* Time-Domain Features:  Mean, Std, Skewness, Kurtosis, Peak2Peak, RMS, CrestFactor,  ShapeFactor, ImpulseFactor, MarginFactor, Energy 
* Frequency-Domain Feature: basic features,  SKMean, SKStd, SKSkewness, SKKurtosis


### Preparation

Download Template Matlab Source file ([download here)](https://github.com/ykkimhgu/digitaltwinNautomation-src/blob/main/Assignment/Assignment_Diagnosis101CWRU/IAIA_Assignment_Diagnosis101_student.mlx)

Download small CWRU Dataset (**[download here](https://github.com/ykkimhgu/digitaltwinNautomation-src/blob/main/Assignment/Assignment_FeatureExtraction_CWRU/Assignment_FeatureExtraction_CWRU_data.zip)**)

* 1 data file for each fault state
* Normal / Outer Race fault / Inner Race fault



## Dataset

### Sampling: 

Drive end bearing(12K): 12,000 samples/second

### Classes:

Normal / Outer Race fault / Inner Race fault
Each under 1HP load,  fault diameter of 0.007inches






## Extract  and Analyze Features   

Extract 

* Time Statistical Features
* Frequency Statistical Features
* Envelop Extraction  Features



Plot

* STFT
* Kurtogram
* Spectral Kurtosis



## Compare Features 







## Report

Submit  in mlx file:  Assignment2_Diagnosis101CWRU_Name.mlx





