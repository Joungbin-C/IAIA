# Dataset



## CWRU dataset

### Introduction 

**CWRU bearing dataset** 
https://engineering.case.edu/bearingdatacenter/download-data-file

**Classes:**

* Normal / Outer Race fault  / Inner Race fault / Ball fault
* Different fault sizes
* Different load
* Drive-End, Fan-End

**Sampling:** 

* Drive end bearing(12K): 12,000 samples/second
* Drive end bearing(48K):  48,000 samples/second  (not used in lecture)
* Fan end bearing data:  12,000 samples/second.  (not used in lecture)

**Variable names**

* DE - drive end accelerometer data
* FE - fan end accelerometer data
* BA - base accelerometer data
* time - time series data
* RPM - rpm during testing



![image](https://github.com/user-attachments/assets/35099f50-ca84-42bd-8219-128680bee37a)




### Dataset folders

**CWRU_selected_dataset**

* Feature_data

  * env_feature
  * stat_feature
  * wpe_feature
  * glob_feature
  * example_data  --> sample_data
  * example_test -->  sample_test
  * example_train --> sample_train

* Raw_data

  * ball_007_1hp  

  * inner_007_1hp

  * normal_1hp  

  * outer_007_1hp

    

**CWRU_full_dataset**

* Raw_data_0hp
  * 12k Drive-End(DE), Fan-End(FE) 
  * Normal, Ball, IR, OR
  * Fault size: 7, 14, 21
  * Load: 0HP

* Raw_data

  * 12k Drive-End(DE), Fan-End(FE) 

  * Normal, Ball, IR, OR

  * Fault size: 7, 14, 21

  * Load: 0HP, 1hp, 3hp
  
  ![image](https://github.com/user-attachments/assets/7addc9d8-e6ae-4de2-94cf-f594e2dd6c32)
  
  



---



## HGU Bearing Dataset

### Introduction 

https://github.com/ykkimhgu/HGU_IAIA/blob/main/HGU%20Bearing%20Dataset%20Description.md



### Dataset 

* HGU bearing dataset v1

[HGU_bearing_dataset_v1.mat](https://drive.google.com/file/d/1bkB45JlS0Z7lILDIBCOj2u4NZzHTqn9i/view?usp=share_link)





---



## PHM Open Dataset 

### Introduction 

https://github.com/ykkimhgu/HGU_IAIA/blob/main/DTA_OpenDataset4PHM.md



### 
