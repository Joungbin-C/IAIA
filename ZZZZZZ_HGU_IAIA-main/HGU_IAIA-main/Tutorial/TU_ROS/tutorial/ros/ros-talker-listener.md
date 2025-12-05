# 노드와 토픽 개념 이해하기

ROS 통신 시스템의 기본 구성요소인 노드(Node)와 토픽(Topic)에 대한 설명을 위한 자료임.

- 실행하기

  ```bash
  roslaunch rospy_tutorials talker_listener.launch
  ```

  

  - talker가 생성한 내용을 listener가 메시지로 받아서 화면에 출력한다.

  ![image](https://user-images.githubusercontent.com/91526930/234394784-a24bfbb2-8f10-443e-b23d-f5dafda2532e.png)





- node & Topic 관찰하기

  ```bash
  rqt_graph
  ```

  

  - talker와 listener라는 노드가 있으며, chatter라는 정보를 전달하고 있다.

  ![image](https://user-images.githubusercontent.com/91526930/234394161-ca099b10-639c-466d-9162-7fe709a4a39a.png)







- Tutorial 폴더 접근하기

  ```bash
  roscd rospy_tutorials
  code .
  ```

  

  - 폴더에는 `.launch` `.py`의 구성

    ![image](https://user-images.githubusercontent.com/91526930/234396103-730b952f-d540-4871-b962-3101a73b3778.png)

  

  - `talker_listener.launch`파일 내에는 두 개의 node를 실행하는 것으로 구성되며, 각각 python 파일로부터 불러와지는 것을 확인할 수 있다.

    ![image](https://user-images.githubusercontent.com/91526930/234396233-154876be-05dc-4bba-b92e-f6e1e1acc233.png)

  

  - `listener.py`

    ![image](https://user-images.githubusercontent.com/91526930/234396748-210f85b3-f6da-42a1-8e1e-434460f27045.png)

    - 'listener'라는 노드를 `init_node`를 통해 초기화한다.

    - 'chatter'라는 String 형태의 msg 정보를 subscribing하고, callback함수를 실행한다. 

    - callback 함수는 data를 받아서, terminal 창에 출력한다.

      

  - `talker.py`

    ![image](https://user-images.githubusercontent.com/91526930/234398302-2ef57b3a-b3d7-4d62-966b-13475a1e5971.png)

    - 'chatter'라는 String 형태의 msg 정보를 publishing하는 변수 pub을 선언한다.
    - 'talker'라는 노드를 초기화한다.
    - hello_str에는 String 형태의 정보를 생산 및 할당한다.
    - publish 함수를 통해 hello_str 변수를 publishing 한다.

    

  - std_msgs

    - [ROS Wiki - std_msgs](http://wiki.ros.org/std_msgs)
    - std_msgs는 기본적으로 정의된 msgs들의 기본 형식을 제공한다. 
    - `Bool`, `Byte`, `Int16`, `Float32`, `Int8MultiArray`, `String`, `Time` 등의 변수 type들이 존재한다.

    ![image](https://user-images.githubusercontent.com/91526930/234399565-051b3c6f-2160-4341-a715-0a4e2f4b68e4.png)



- 노드와 메세지의 활용법을 더 자세히 알고자 한다면, 002~009의 tutorial을 통해서 연습하도록 합시다.
