# ROS 설치

Reference: [ROS wiki - Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)



## 1. Setup sources & keys

```bash
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```

&nbsp;

## 2. Installation

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full
```

&nbsp;

## 3. Environment setup

```bash
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

&nbsp;

## 4. Dependencies for building packages

```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

&nbsp;

## 5. Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

&nbsp;

## 6. Create a ROS Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```



