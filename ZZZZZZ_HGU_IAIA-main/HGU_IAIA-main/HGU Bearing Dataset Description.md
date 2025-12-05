# HGU Bearing Dataset



## Description

This is a small-sized dataset of ball bearing accelerations for normal and faulty bearings.

Experiments were conducted on the lab-manufactured testbench using a 1.4kW Electric motor.  



![](https://user-images.githubusercontent.com/84221531/208964624-de3da554-c9ac-42ee-81f6-416f54acf72c.png)

![](https://user-images.githubusercontent.com/82484459/231032403-e408eb77-b7a8-4177-abde-a3646b4621d7.png)

The test bench for motor performance assessment consists of:

- AC Servo Motor with 1.4 kW (HG-KR43J)

- Gearbox 1:100 

- Gearbox 50:1

- Torque transducer (UTM2-10Nm)

- Accelerometer (PCB-(M)352C66)

  

### Bearing Faults

The test bearings support the motor shaft. Defects were introduced at a single point by EDM machining. The diameters of defects in millimeters:

-  **Inner, outer race fault size**: 
   -  diameter 1 mm / depth 2.3 mm
-  **Ball fault size**: 
   -  diameter 1 mm / depth 1 mm

**Bearing Info:**
* D1 = 37.05
* D2 = 24.95
* N_B = 13
* Beta = 6.05

### Other Faults

The testbench can also apply shaft misalignment faults and load imbalance faults

* **Misalignment Fault:** 
  * Aligned
  * Misaligned: 1mm, 2mm
* **Imbalance Fault:**
  * Balanced disk
  * Imbalanced disk




##  Dataset File

**Dataset download**: [HGU_bearing_dataset_v1.mat](https://drive.google.com/file/d/1bkB45JlS0Z7lILDIBCOj2u4NZzHTqn9i/view?usp=share_link) 

****Data was collected for normal bearings, defect bearing, and defect bearing with misalignment. Data were collected at a sampling rate of 12,000 samples/second for 10 seconds. Data files are in **Matlab** format.

This dataset corresponds to the following conditions:

- Aligned shaft / Misaligned shaft

- Shaft rotating speed of 500 rpm

- 12 kHz sampling frequency of the accelerometers


### 

### Fault Class

The bearing health conditions are classified as follows:

| Fault type       | Aligned shaft | Misaligned shaft |
| ---------------- | ------------- | ---------------- |
| Normal           | N1            | N2               |
| Inner race fault | I1            | I2               |
| Outer race fault | O1            | O2               |
| Ball fault       | B1            | B2               |

Each data sample was obtained using a window with a size of 12000 and 1000 strides for data augmentation. 



**bearing_acc**

* raw acceleration data captured for 10 secs

**trainset**

* train data : 87*8=696 datasets

**testset**

* test data: 21*8=168 datasets




A total of 108 data samples were obtained for each class, and the ratio of train/test dataset was set to 8:2 for validation. 

 * 87 train datasets and 21 test datasets were obtained from randomly 108 data samples for each class. 



| Class | Train samples | Test samples |
| ----- | ------------- | ------------ |
| N1    | 87            | 21           |
| N2    | 87            | 21           |
| I1    | 87            | 21           |
| I2    | 87            | 21           |
| O1    | 87            | 21           |
| O2    | 87            | 21           |
| B1    | 87            | 21           |
| B2    | 87            | 21           |
| Sum   | 696           | 168          |

