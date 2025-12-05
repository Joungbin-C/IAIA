## Moveit Tutorial

- reference: 
  - [Moveit - Installation](https://ros-planning.github.io/moveit_tutorials/doc/getting_started/getting_started.html)
  - [Moveit - Quick Start in Rviz](https://ros-planning.github.io/moveit_tutorials/doc/quickstart_in_rviz/quickstart_in_rviz_tutorial.html)
  - [Moveit - Python Interface](https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html)



## Getting Started

### Install ROS and Catkin

```bash
rosdep update
sudo apt update
sudo apt dist-upgrade

sudo apt install ros-noetic-catkin python3-catkin-tools python3-osrf-pycommon
sudo apt install python3-wstool
```

### Download MoveIt Source

```bash
[위치] ~/catkin_ws/src
wstool init .
wstool merge -t . https://raw.githubusercontent.com/ros-planning/moveit/master/moveit.rosinstall
wstool remove  moveit_tutorials  # this is cloned in the next section
wstool update -t .
```

### Download Example Code

```bash
[위치] ~/catkin_ws/src
git clone https://github.com/ros-planning/moveit_tutorials.git -b master
git clone https://github.com/ros-planning/panda_moveit_config.git -b noetic-devel
```

### Build

```bash
[위치] ~/catkin_ws/src
rosdep install -y --from-paths . --ignore-src --rosdistro noetic

[위치] ~/catkin_ws
catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build		# catkin_make 기록이 있어서 실행이 안된다고 하면, cakin clean으로 devel과 build 폴더 정리. 그 이후에 build 가능함.

source ~/catkin_ws/devel/setup.bash
```



## Quick Start in Rviz

### Launch the demo

```bash
roslaunch panda_moveit_config demo.launch rviz_tutorial:=true
```

- [Moveit - Quick Start in Rviz](https://ros-planning.github.io/moveit_tutorials/doc/quickstart_in_rviz/quickstart_in_rviz_tutorial.html) 에서 실습할 것.



## Move Group Python Interface

### Start RViz and MoveGroup node

```bash
roslaunch panda_moveit_config demo.launch
rosrun moveit_tutorials move_group_python_interface_tutorial.py
```

- 파이썬 파일을 실행시키면 'Enter'를 누르면서, 동작하는 것을 확인할 수 있음.

- 파일 위치: `moveit_tutorials/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py`

- nodes와 topics

  ![image](https://user-images.githubusercontent.com/91526930/235368410-33cbeb5d-e844-40d1-b679-2d5e700a736d.png)

  ![image](https://user-images.githubusercontent.com/91526930/235368537-7688202e-65a3-4bb9-aca8-e32d3a2f72de.png)





