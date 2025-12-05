# 로봇 구동
## 1. [로봇 연결](robot-connection.md)

&nbsp;
## 2. 데모 수행

### Demo 1: Simple Move
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2
roslaunch ur5e_rg2_moveit_config move_group.launch
rosrun ur_python demo_move.py
```


### Demo 2: Gripper Operation
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2
roslaunch ur5e_rg2_moveit_config move_group.launch
rosrun ur_python demo_grip.py
```


### Demo 3: Pick & Place
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2
roslaunch ur5e_rg2_moveit_config move_group.launch
rosrun ur_python demo_pick_and_place.py
```

### Demo 4: Pet Feeder Robot
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2
roslaunch ur5e_rg2_moveit_config move_group.launch
rosrun ur_python camera.py
rosrun ur_python image_display.py
rosrun ur_python pet_classifier.py    # conda environment
rosrun ur_python pet_feeder.py
```
