# ROS Packages for Robots

## Moveit



```bash
sudo apt-get install ros-noetic-moveit
sudo apt-get install ros-noetic-industrial-core
sudo apt-get install ros-noetic-moveit-visual-tools
sudo apt-get install ros-noetic-joint-state-publisher-gui
```



## Gazebo



```bash
sudo apt-get install ros-noetic-gazebo-ros-pkgs
sudo apt-get install ros-noetic-gazebo-ros-control
sudo apt-get install ros-noetic-joint-state-controller
sudo apt-get install ros-noetic-effort-controllers
sudo apt-get install ros-noetic-position-controllers
sudo apt-get install ros-noetic-joint-trajectory-controller
```







## UR Robots

Reference: [Github - universal robot](https://github.com/ros-industrial/universal_robot)

```bash
sudo apt-get install ros-noetic-universal-robots
```



```bash
cd ~/catkin_ws/src

# retrieve the sources
git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git

cd ~/catkin_ws

# checking dependencies
rosdep update
rosdep install --rosdistro noetic --ignore-src --from-paths src

# building
catkin_make

# activate this workspace
source ~/catkin_ws/devel/setup.bash
```



* demo.launch

```bash
roslaunch ur5e_moveit_config demo.launch
```



## UR Robot Driver

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



## other packages

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

(rg2_description is from https://github.com/ekorudiawan/rg2_simulation)

copy & paste packages about indy10, ur5e, and rg2 gripper in `~/catkin_ws/src`

```bash
cd ~/catkin_ws

# build the workspace
catkin_make

# activate the workspace (ie: source it)
source devel/setup.bash
```


