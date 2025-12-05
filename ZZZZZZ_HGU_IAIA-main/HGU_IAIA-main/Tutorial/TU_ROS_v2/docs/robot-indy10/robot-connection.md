# Indy10 Connection
## Required Packages
아래의 패키지 내역이 `catkin_ws/src` 내에 포함되어 있는지 확인
- `indy_driver`
- `indy_utils`
- `indy10_control`
- `indy10_description`
- `indy10_gazebo`
- `indy10_moveit_config`

패키지가 없으면, [Github-Industrial-AI-Automation_HGU](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Tutorial/TU_ROS/tutorial/ros/ros-install-packages-for-robot.md) 자료의 Ohter packages 내용 참고해서 다운.

&nbsp;
## Execution Flow
## 1. Indy10 Hardware Connection
### 1) 로봇 전원 켜기
  - 전원버튼을 누르면, 약 3분 동안 부팅과정을 거친다. 


### 2) PC의 WiFi를 `HGU_AIAI-5G`로 연결
   - pw: `nth115!!`


### 3) 로봇 IP 주소 확인
   - No.1: 192.168.0.8
   - No.2: 192.168.0.9

&nbsp;
## 2. Software Connection
### Terminal
```bash
roslaunch indy10_moveit_config moveit_planning_execution.launch robot_ip:=192.168.0.8
```

&nbsp;
## 3. gripper/vacuum 사용 설정

### `demo_grip.py`
```python
indy10 = MoveGroupPythonInterface(real=True, gripper="Vacuum")
```
### `move_group_python_interface.py` 내 클래스 참조
```python
real = True
gripper = "Gripper" or "Vaccum"
```

&nbsp;
## 4. 시뮬레이션 실행(gazebo)
### Terminal
```bash
roslaunch indy10_gazebo indy10_moveit_gazebo.launch
```

&nbsp;
## 5. 데모 프로그램 실행
### Terminal
```bash
rosrun indy_driver demo_move.py
```



