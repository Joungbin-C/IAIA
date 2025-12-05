# Simulation - Indy10



### Demo 1: Simple Move

```bash
roslaunch indy10_gazebo indy10_moveit_gazebo.launch
rosrun indy_driver demo_move.py
```

- `move_group_python_interface.py` 수정하기: [소스코드](https://github.com/ykkimhgu/HGU_IAIA/blob/main/Tutorial/TU_ROS/packages/indy_driver/src/move_group_python_interface.py)


- `demo_move.py`

  ```python
  #!/usr/bin/env python3
  #-*- coding:utf-8 -*- 
  
  from move_group_python_interface import MoveGroupPythonInterface
  from math import tau
  import rospy
  
  def main():
      try:
  
          indy10 = MoveGroupPythonInterface(real=False)
          
          print("============ Moving initial pose using a joint state goal ...")
          # indy10.move_to_standby()
          init_pose_joints = [0, 0, tau/4, 0, tau/4, 0]          # tau = 2 * pi
          indy10.go_to_joint_abs(init_pose_joints)
  
          input("============ Press `Enter` to execute a movement using a joint state goal ...")
          joint_rel = [1/8 * tau, 0, 0, 0, 0, 0]          # tau = 2 * pi
          indy10.go_to_joint_rel(joint_rel)
          
          input("============ Press `Enter` to execute a movement using a joint state goal ...")
          joint_rel = [-1/8 * tau, 0, 0, 0, 0, 0]          # tau = 2 * pi
          indy10.go_to_joint_rel(joint_rel)
  
          input("============ Press `Enter` to execute a movement using a relative pose ...")
          target_pose_rel_xyz = [0.0, 0.0, 0.0]
          target_pose_rel_rpy = [tau/8, 0, 0]
          indy10.go_to_pose_rel(target_pose_rel_xyz, target_pose_rel_rpy)
          
          
          input("============ Press `Enter` to execute a movement using a relative pose ...")
          target_pose_rel_xyz = [0.0, 0.0, 0.0]
          target_pose_rel_rpy = [-tau/8, 0, 0]
          indy10.go_to_pose_rel(target_pose_rel_xyz, target_pose_rel_rpy)
  
          
          input("============ Press `Enter` to execute a movement using a absolute pose ...")
          target_pose_abs_xyz = [0.0, 0.52, 0.54]
          target_pose_abs_rpy = [-2.3553, 0.0003, 0.0008]
          indy10.go_to_pose_abs(target_pose_abs_xyz, target_pose_abs_rpy)
  
          print("============ complete!")
  
      except rospy.ROSInterruptException:
          return
      except KeyboardInterrupt:
          return
  
  if __name__ == "__main__":
      main()
  ```




### Demo 2: Pick & Place

```bash
roslaunch indy10_gazebo indy10_moveit_gazebo.launch
rosrun indy_driver demo_pick_and_place.py
```

- `demo_pick_and_place.py`
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
  
  DEG2RAD = pi/180
  RAD2DEG = 180/pi
  
  def main():
      try:
          print("grip")
          indy10 = MoveGroupPythonInterface(real="sim")
          
          indy10.move_to_standby()
          # indy10.grip_off()
          
          # go to A top
          pos_A = np.array([-45.0, 0.0, 90.0, 0.0, 90.0, 0.0])*DEG2RAD
          indy10.go_to_joint_abs(pos_A)
  
          # go to A bottom
          rel_xyz = [0.0, 0.0, -0.22]
          rel_rpy = [0.0, 0.0, 0.0]
          indy10.go_to_pose_rel(rel_xyz, rel_rpy)
  
          # grip
          # indy10.grip_on()
  
          # go to A top
          rel_xyz = [0.0, 0.0, 0.22]
          rel_rpy = [0.0, 0.0, 0.0]
          indy10.go_to_pose_rel(rel_xyz, rel_rpy)
  
  
          # go to B top
          pos_B = np.array([45.0, 0.0, 90.0, 0.0, 90.0, 0.0])*DEG2RAD 
          indy10.go_to_joint_abs(pos_B)
  
          # go to B bottom
          rel_xyz = [0.0, 0.0, -0.22]
          rel_rpy = [0.0, 0.0, 0.0]
          indy10.go_to_pose_rel(rel_xyz, rel_rpy)
  
          # grip
          # indy10.grip_off()
  
          # go to B top
          rel_xyz = [0.0, 0.0, 0.22]
          rel_rpy = [0.0, 0.0, 0.0]
          indy10.go_to_pose_rel(rel_xyz, rel_rpy)
  
          indy10.move_to_standby()
  
          print("complete")
      
      except rospy.ROSInterruptException:
          return
      except KeyboardInterrupt:
          print("Shut down by Key Interrupt")
          return
  
  
  if __name__ == '__main__':
      main()
  ```
  
### Demo 3: Pet Feeder Robot

```bash
roslaunch indy10_gazebo indy10_moveit_gazebo.launch
rosrun tutorial camera.py
rosrun tutorial image_display.py
rosrun indy_driver pet_classifier.py
rosrun indy_driver pet_feeder.py
```




