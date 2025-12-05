# ROS 실행하기

turtlesim 예제를 통해, 설치한 ROS를 실행해보기

- Ros master 실행

  ```bash
  roscore
  ```

  

- Turtlesim node

  ```bash
  rosrun turtlesim turtlesim_node
  ```

  거북이가 새로운 창에 출력된다.

  

- teleop key

  ```bash
  rosrun turtlesim turtle_teleop_key
  ```

  방향키를 누르면, 거북이가 움직인다.



- node & Topic 관찰하기

  ```bash
  rqt_graph
  ```


