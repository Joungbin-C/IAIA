# ROS Packages for Robots

- 시뮬레이션(MoveIt, Gazebo 등)과 산업용 로봇인 UR Robot, 카메라 관련 패키지들을 ROS Noetic 버전에 맞게 설치하는 명령어들을 기록함.
- 프로젝트 진행에 기본적으로 설치 요구되는 패키지들이며, 패키지 설치/복사 등 자세한 설명은 ROS 기본 튜토리얼을 참고할 것.

- Simulation (MoveIt, Gazebo, etc.), industrial robots such as UR Robot, and camera-related packages are recorded here with installation commands for ROS Noetic.

- These packages are required as a baseline for project development. For detailed explanations on package installation/copying, refer to the ROS Basic Tutorial.


&nbsp;
## 1. MoveIt


```bash
sudo apt-get install ros-noetic-moveit
sudo apt-get install ros-noetic-industrial-core
sudo apt-get install ros-noetic-moveit-visual-tools
sudo apt-get install ros-noetic-joint-state-publisher-gui
```


&nbsp;
## 2. Gazebo

```bash
sudo apt-get install ros-noetic-gazebo-ros-pkgs
sudo apt-get install ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-joint-state-controller
sudo apt-get install ros-noetic-effort-controllers
sudo apt-get install ros-noetic-position-controllers
sudo apt-get install ros-noetic-joint-trajectory-controller
```


&nbsp;
## 3. UR Robot

Reference: [Github - universal robot](https://github.com/ros-industrial/universal_robot)

### Terminal
```bash
sudo apt-get install ros-noetic-universal-robots

cd ~/catkin_ws/src

# retrieve the sources
git clone -b noetic-devel https://github.com/ros-industrial/universal_robot.git

cd ~/catkin_ws

# checking dependencies
rosdep update
rosdep install --rosdistro noetic --ignore-src --from-paths src

# building
catkin_make

# activate this workspace
source ~/catkin_ws/devel/setup.bash
```

### Demo Test: `demo.launch`

```bash
roslaunch ur5e_moveit_config demo.launch
```


&nbsp;
## 4. UR Robot ROS Driver

Reference: [Github - universal robot ROS driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)

```bash
cd ~/catkin_ws/src

# retrieve the sources
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git

cd ~/catkin_ws

# install dependencies
sudo apt update -qq
rosdep update
rosdep install --from-paths src --ignore-src -y

# build the workspace
catkin_make

# activate the workspace (ie: source it)
source devel/setup.bash
```


&nbsp;
## 5. USB-Camera

```bash
sudo apt-get install ros-noetic-cv-bridge
sudo apt install ros-noetic-usb-cam
sudo apt-get update
```


&nbsp;
## 6. Intel RealSense2 Camera

- reference: [Github - IntelRealSense(ros1-legacy)](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy)
- Driver 설치

```bash
sudo apt-get install ros-noetic-realsense2-camera
sudo apt-get install ros-noetic-realsense2-description
```


&nbsp;
## 7. 실습용 패키지 (별도 제공) Paractive Packages(Provided Separately)

### 1) 배경화면에 폴더 `packages_from_git` 생성 및 해당 폴더로 위치 이동
### 1) Create a folder `packages_from_git` on the Desktop adn navigate into it
```bash
cd ~/Desktop
mkdir packages_from_git
cd packages_from_git
```
&nbsp;
### 2) git 명령어 활용하여 `ykkimhgu/HGU_IAIA.git`에서 제공하는 실습을 위해 제공되는 패키지 가져오기
### 2) Use git commands to fetch to practice packages provided from `ykkimhgu/HGU_IAIA.git`
```bash
cd ~/Desktop
mkdir packages_from_git
cd packages_from_git

git init
git config core.sparseCheckout true
git remote add -f origin https://github.com/ykkimhgu/HGU_IAIA.git
echo "Tutorial/TU_ROS/packages" >> .git/info/sparse-checkout
git pull origin main
```
&nbsp;
### 3) 가져온 패키지들은 `packages_from_git`에 저장되며, 해당 패키지들을 `~/catkin_ws/src`에 복사/붙여넣기
### 3) The retrieved packages will be stored in packages_from_git. Copy them into `~/catkin_ws/src`.

&nbsp;
### 4) 빌드 실행하기
### 4) Build the workspace
```bash
cd ~/catkin_ws

# build the workspace
catkin_make

# activate the workspace (ie: source it)
source devel/setup.bash
```



## 검토: 패키지 목록 
## Verification: Package List

```bash
catkin_ws
└── build
└── devel
└── src 
    ├── indy10_control
    ├── indy10_description
    ├── indy10_moveit_config
    ├── indy_driver
    ├── rg2_description
    ├── universal_robot
    ├── Universal_Robots_ROS_Driver
    ├── ur5e_rg2_moveit_config
    └── ur_python
```



