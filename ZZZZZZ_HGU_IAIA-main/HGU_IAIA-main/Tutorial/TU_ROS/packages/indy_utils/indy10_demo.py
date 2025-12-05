#!/usr/bin/env python3
#-*- coding:utf-8 -*- 

import indydcp_client as client
import copy


def main():
    robot_ip = "192.168.0.8"    # 예시 STEP IP 주소
    robot_name = "NRMK-Indy10"   # IndyRP2의 경우 "NRMK-IndyRP2"
    indy = client.IndyDCPClient(robot_ip, robot_name) # indy 객체 생성

    indy.connect() # 로봇 연결

    # Setting 
    indy.set_joint_vel_level(9)     # 1 ~ 9
    indy.set_task_vel_level(9)     # 1 ~ 9

    # Robot Status
    status = indy.get_robot_status()
    print("[robot status]:", status)

    # Get Joint Pose
    joint_pos = indy.get_joint_pos() # [q1, q2, q3, q4, q5, q6]
    print("[current joint]:", joint_pos)

    # Move Joint to Absolute value
    targ_joint = copy.deepcopy(joint_pos)
    targ_joint[0] = joint_pos[0] - 45
    print("[target joint]:", targ_joint)
    indy.joint_move_to(targ_joint)
    
    # Wait
    status = indy.get_robot_status()
    while status['busy']:
        print(indy.get_joint_vel())
        status = indy.get_robot_status()
    
    print("[current joint]: ", indy.get_joint_pos()) # [q1, q2, q3, q4, q5, q6]

    # Move Joint to Relative value
    indy.joint_move_by([45, 0, 0, 0, 0, 0])
    
    # Wait
    status = indy.get_robot_status()
    while status['busy']:
        status = indy.get_robot_status()

    print("[current joint]: ", indy.get_joint_pos()) # [q1, q2, q3, q4, q5, q6]

    # task position to Relative value
    print("[task] :", indy.get_task_pos()) # [x, y, z, u, v, w]
    indy.task_move_by([0, -0.1, 0, 0, 0, 0])
    
    # Wait
    status = indy.get_robot_status()
    while status['busy']:
        print(indy.get_task_vel())
        status = indy.get_robot_status()

    print("[task] :", indy.get_task_pos()) # [x, y, z, u, v, w]
    indy.task_move_by([0, 0.1, 0, 0, 0, 0])

    # Wait
    status = indy.get_robot_status()
    while status['busy']:
        status = indy.get_robot_status()

    print("[task] :", indy.get_task_pos()) # [x, y, z, u, v, w]

    indy.disconnect() # 연결 해제


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print("[ERROR]", e)
