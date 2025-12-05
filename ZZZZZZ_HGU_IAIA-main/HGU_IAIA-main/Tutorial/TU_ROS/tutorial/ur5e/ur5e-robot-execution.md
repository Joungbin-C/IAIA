# Robot Execution - UR5e

## Required Packages

아래의 패키지 내역이 `catkin_ws/src` 내에 포함되어 있는지 확인하세요.

- `rg2_description`
- `universal_robot`
- `Universal_Robots_ROS_Driver`
- `ur_python`
- `ur5e_rg2_moveit_config`

패키지가 없으면, [Github-Industrial-AI-Automation_HGU](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Tutorial/TU_ROS/tutorial/ros/ros-install-packages-for-robot.md) 자료의 Ohter packages 내용 참고해서 다운받아 주시면 됩니다.


## UR5e Robot Execution

### Robot Setting

1) 티칭 펜던트의 전원을 켠다.
2) 사전에 작업된 `ros_rg2.urp` 로봇 프로그램을 불러온다.
3) 로봇의 전원을 켠다.
4) 컨트롤 박스의 랜선을 PC에 연결한다.
5) PC에서 유선 네트워크 연결을 설정한다.

더 자세한 내용은 [Github-Setting UR5e](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Tutorial/TU_ROS/tutorial/ur5e/ur5e-setting.md)에서 확인한다.


### Execution Flow

- **Hardware Connection (Robot Setting 참고)**

- **Software Connection**

  ```bash
  roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.0.2
  ```

- **로봇 프로그램 실행 (Teaching Pendant)**

  <img src="https://user-images.githubusercontent.com/91526930/234138529-75eb185e-f308-400f-aebb-d2f79e8b3ffb.png" alt="image" style="zoom:70%;" />

- **Moveit**

  ```bash
  roslaunch ur5e_rg2_moveit_config move_group.launch
  ```

- **Demo Program**

  ```bash
  rosrun ur_python demo_move.py
  ```

  

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

- `pet_feeder.py` 수정
  ```python
  #!/usr/bin/env python3
  #-*- coding:utf-8 -*-
  
  import sys
  import rospy
  import moveit_commander
  import moveit_msgs
  import geometry_msgs
  
  import tf
  import numpy as np
  
  from math import pi
  
  from move_group_python_interface import MoveGroupPythonInterface
  from ur_python.msg import pet_info
  
  DEG2RAD = pi/180
  RAD2DEG = 180/pi
  
  
  feed_info = {   "pt_A_top"   :   {  "tabby cat"         : np.array([30.16, -95.54, 94.27, -88.67, -89.83, -57.71])*DEG2RAD   ,
                                      "Golden Retriever"  : np.array([30.16, -95.54, 94.27, -88.67, -89.83, -57.71])*DEG2RAD   },
                  "pt_A_bottom":   {  "tabby cat"         : np.array([30.13, -87.14, 129.82, -122.63, -89.81, -57.64])*DEG2RAD  ,
                                      "Golden Retriever"  : np.array([30.13, -87.14, 129.82, -122.63, -89.81, -57.64])*DEG2RAD  },
                  "pt_B_top"   :   {  "tabby cat"         : np.array([112.29, -91.15, 92.10, -90.90, -89.82, 19.23])*DEG2RAD    ,
                                      "Golden Retriever"  : np.array([112.29, -91.15, 92.10, -90.90, -89.82, 19.23])*DEG2RAD    },
                  "pt_B_bottom":   {  "tabby cat"         : np.array([112.26, -85.02, 112.84, -117.77, -89.80, 19.29])*DEG2RAD   ,
                                      "Golden Retriever"  : np.array([112.26, -85.02, 112.84, -117.77, -89.80, 19.29])*DEG2RAD   },
                  
                  "quantity"  :   {   "tabby cat"         : 2,
                                      "Golden Retriever"  : 4},
  
                  "up_A"       :   {  "rel_xyz"           : [0.0, 0.0, 0.213]     ,
                                      "rel_rpy"           : [0.0, 0.0, 0.0]      },
                  "down_A"     :   {  "rel_xyz"           : [0.0, 0.0, -0.213]    ,
                                      "rel_rpy"           : [0.0, 0.0, 0.0]      },
                  "up_B"       :   {  "rel_xyz"           : [0.0, 0.0, 0.178]     ,
                                      "rel_rpy"           : [0.0, 0.0, 0.0]      },
                  "down_B"     :   {  "rel_xyz"           : [0.0, 0.0, -0.178]    ,
                                      "rel_rpy"           : [0.0, 0.0, 0.0]       }
              }
  
  
  class PetFeederNode():
      def __init__(self):
          # Subscriber
          self.sub_pet_class = rospy.Subscriber("pet_classifier/pet_info", pet_info, self.feed)  # camera/image_raw 토픽에서 Image 메시지 수신
  
          # Robot initialization
          self.robot = MoveGroupPythonInterface(real="real")
          self.robot.move_to_standby()
          self.robot.grip_off()
      
      def feed(self, pet_info):
          
          for i in range(feed_info["quantity"][pet_info.name]):
              
              # point A
              # go_to_pose_abs(self, absolute_xyz, absolute_rpy)
              self.robot.go_to_joint_abs(feed_info["pt_A_top"][pet_info.name])
              self.robot.go_to_pose_rel(feed_info["down_A"]['rel_xyz'], feed_info["down_A"]['rel_rpy'])
              self.robot.grip_on()
              self.robot.go_to_pose_rel(feed_info["up_A"]['rel_xyz'], feed_info["up_A"]['rel_rpy'])
  
              # point B
              self.robot.go_to_joint_abs(feed_info["pt_B_top"][pet_info.name])
              self.robot.go_to_pose_rel(feed_info["down_B"]['rel_xyz'], feed_info["down_B"]['rel_rpy'])
              self.robot.grip_off()
              self.robot.go_to_pose_rel(feed_info["up_B"]['rel_xyz'], feed_info["up_B"]['rel_rpy'])
  
          self.robot.move_to_standby()
              
      def run(self):
          rospy.spin()                                    # 노드가 종료될 때까지 계속 실행
  
  if __name__ == '__main__':
      try:
          pet_feeder = PetFeederNode()
          pet_feeder.run()                # run 메서드 실행
      except rospy.ROSInterruptException:
          pass
  ```
