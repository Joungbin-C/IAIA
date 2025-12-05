#!/usr/bin/env python3
#-*- coding:utf-8 -*- 
import rospy
import numpy as np
from move_group_python_interface import MoveGroupPythonInterface
from math import tau

DEG2RAD = tau / 360.0

def main():
    try:

        ur5e = MoveGroupPythonInterface(real="sim")

        print("============ Moving initial pose using a joint state goal ...")
        # ur5e.move_to_standby()
        init_pose_joints = [tau/4, -tau/4, tau/4, -tau/4, -tau/4, 0.0]          # tau = 2 * pi
        ur5e.go_to_joint_abs(init_pose_joints)

        input("============ Press `Enter` to execute a movement using a joint state goal(relative) ...")
        joint_rel = [1/8 * tau, 0, 0, 0, 0, 0]          # tau = 2 * pi
        ur5e.go_to_joint_rel(joint_rel)
        
        input("============ Press `Enter` to execute a movement using a joint state goal(relative) ...")
        joint_rel = [-1/8 * tau, 0, 0, 0, 0, 0]          # tau = 2 * pi
        ur5e.go_to_joint_rel(joint_rel)

        input("============ Press `Enter` to execute a movement using a relative pose ...")
        target_pose_rel_xyz = [0.0, 0.0, 0.0]
        target_pose_rel_rpy = [tau/8, 0, 0]
        ur5e.go_to_pose_rel(target_pose_rel_xyz, target_pose_rel_rpy)
                
        input("============ Press `Enter` to execute a movement using a relative pose ...")
        target_pose_rel_xyz = [0.0, 0.0, 0.0]
        target_pose_rel_rpy = [-tau/8, 0, 0]
        ur5e.go_to_pose_rel(target_pose_rel_xyz, target_pose_rel_rpy)
        
        input("============ Press `Enter` to execute a movement using a absolute pose ...")
        target_pose_abs_xyz = [-0.13, 0.49, 0.47]
        target_pose_abs_rpy = [-2.3939, -0.00, -0.00]
        ur5e.go_to_pose_abs(target_pose_abs_xyz, target_pose_abs_rpy)

        
        print("============ complete!")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        print("Shut down by Key Interrupt")
        return

if __name__ == "__main__":
    main()
