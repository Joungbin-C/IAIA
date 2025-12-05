# 로봇 구동
## 1. [로봇 연결](robot-connection.md)

&nbsp;
## 2. 데모 수행

### Demo 1: Simple Move

```bash
roslaunch indy10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.0.8
rosrun indy_driver demo_move.py
```



### Demo 2: Gripper Operation

```bash
roslaunch indy10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.0.8
rosrun indy_driver demo_grip.py
```




### Demo 3: Pick & Place

```bash
roslaunch indy10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.0.8
rosrun indy_driver demo_pick_and_place.py
```

### Demo 4: Pet Feeder Robot

- LAB 수행: [link](https://github.com/ykkimhgu/HGU_IAIA/blob/main/IAIA_LAB_PetFeederRobot.md)
