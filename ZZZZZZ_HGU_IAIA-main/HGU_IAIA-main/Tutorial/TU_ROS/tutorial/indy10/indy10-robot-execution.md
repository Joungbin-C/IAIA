# Robot Execution - Indy10

## Required Packages

아래의 패키지 내역이 `catkin_ws/src` 내에 포함되어 있는지 확인하세요.

- `indy_driver`
- `indy_utils`
- `indy10_control`
- `indy10_description`
- `indy10_gazebo`
- `indy10_moveit_config`

패키지가 없으면, [Github-Industrial-AI-Automation_HGU](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Tutorial/TU_ROS/tutorial/ros/ros-install-packages-for-robot.md) 자료의 Ohter packages 내용 참고해서 다운받아 주시면 됩니다.



## Indy10 Robot Execution

### Robot Setting

1) 로봇의 전원을 켠다.
2) PC의 WiFi를 sslab으로 연결한다.
   - pw: `nth115!!`
3) 로봇의 IP 주소를 확인한다.
   - No.1: 192.168.0.8
   - No.2: 192.168.0.9



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

- gripper 사용 설정

   - `demo_grip.py`
  
      ```python
      indy10 = MoveGroupPythonInterface(real=True, gripper="Vaccum")
      ```
   - `move_group_python_interface.py` 내 클래스 참조
      ```python
      real = True
      gripper = "Gripper" or "Vaccum"
      ```


### Demo 3: Pick & Place

```bash
roslaunch indy10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.0.8
rosrun indy_driver demo_pick_and_place.py
```

### Demo 4: Pet Feeder Robot

- LAB 수행: [link](https://github.com/ykkimhgu/HGU_IAIA/blob/main/IAIA_LAB_PetFeederRobot.md)


