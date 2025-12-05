# Anaconda & Deep Learning Framework



## 1. Anaconda Installation

### Anaconda Installer

- Download anaconda Installer: [link](https://www.anaconda.com/download/success)



### Execute Installer File (shell script)

```bash
cd ~/Downloads			  # 다운로드 위치로 이동
bash Anaconda3-2022.10-Linux-x86_64.sh	# shell script 실행
```

- 프로그램 설치 약관에 대한 내용 출력시, Enter를 계속해서 누르고 있으면 됨
- 묻는 질문에는 모두 yes 입력

[reference link](https://record-everything.tistory.com/entry/Ubuntu-2004-%EC%9A%B0%EB%B6%84%ED%88%AC%EC%97%90-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%EC%84%A4%EC%B9%98-%EB%B0%8F-Python-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95)



### Initialize bashrc 

```bash
source ~/.bashrc
```

- 프롬프트를 실행할 때마다 디폴트로 conda 환경이 자동으로 활성화되면서, 터미널에 항상 `(base)`가 함께 출력됨

- 터미널을 열때마다 conda 환경을 활성화되는 것을 막기 위해 아래 명령어 입력

```bash
conda config --set auto_activate_base false
```



## 2. Create Environment

```bash
# Update CONDA in Base
conda update -n base -c defaults conda

# Create myEnv=py38
conda create -n py38 python=3.8.10

# Activate myEnv
conda activate py38

# Install Numpy, OpenCV, Matplot, Jupyter
conda install -c anaconda seaborn jupyter
pip install opencv-python
```



## 3. Install Deep learning Framework

### With GPU

```bash
# Check GPU model

# Install NVIDIA Driver from Website

# Install CUDA and cuNN
conda install -c anaconda cudatoolkit=11.8 cudnn 

# Install PyTorch
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torchsummary

# Check Installed Packaged in myENV
conda list all
```



### Only CPU

```bash
# CPU Only - PyTorch 2.1
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 cpuonly -c pytorch
pip install torchsummary
```



## 4. ROS 패키지 설치 및 환경구축

- Conda 가상환경에 ROS 패키지 설치
  ```bash
  # Activate myEnv
  conda activate py38
  
  # install rospkg, catkin_pkg
  conda install rospkg catkin_pkg
  ```

- 환경변수: ROS python 라이브러리 경로 추가

  ```bash
  # 편집기 실행
  gedit ~/.bashrc

  # 환경변수 추가
  export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
  ```


## 5. 에러 
### Libtiff 버전 호환 문제
- 상황: 위의 설치과정 이후, `ex_pet_classifier.py` 예제 실행 시, 아래 메세지 출력됨.
- 에러 메시지: `/lib/libgdal.so.26: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0`
- conda env에 `libtiff`의 버전이 적절하지 않아 발생하는 문제임.
  
  ```bash
  # 현재 설치된 libtiff의 버전 확인하기
  conda list libtiff

  # libtiff 4.0으로 설치하기
  conda install -c conda-forge libtiff=4.0
  ```
