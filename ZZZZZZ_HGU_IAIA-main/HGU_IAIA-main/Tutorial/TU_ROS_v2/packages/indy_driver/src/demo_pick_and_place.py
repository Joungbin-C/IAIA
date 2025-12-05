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