# 노드와 토픽의 개념 및 실습

### 노드와 토픽이란 무엇인가?

#### **노드(Node)**

- ROS에서 실행 가능한 프로그램 단위로, 특정 작업(예: 센서 데이터 수집, 모터 제어 등)을 수행.
- 각 노드는 독립적으로 실행되며, 다른 노드와 데이터를 주고받기 위해 통신합니다.



#### **토픽(Topic)**

- 노드 간 데이터를 송수신하는 **비동기 통신 채널**.
- 퍼블리셔(Publisher)와 서브스크라이버(Subscriber)를 통해 데이터가 교환됩니다.
- 예: 카메라 노드가 `/camera/image_raw` 토픽을 통해 이미지 데이터를 퍼블리싱하고, 이를 처리하는 노드가 해당 데이터를 서브스크라이브.



### 실습: TurtleSim

#### 1. ROS 노드 실행

1. **ROS 마스터 실행**: 모든 ROS 통신의 중심이 되는 `roscore` 실행.

   ```
   roscore
   ```

2. **Turtlesim 노드 실행**:

   ```
   rosrun turtlesim turtlesim_node
   ```

   - GUI 창이 열리며 거북이가 나타납니다.

3. **Turtlesim 제어 노드 실행**:

   ```
   rosrun turtlesim turtle_teleop_key
   ```

   - 키보드를 사용하여 거북이를 이동시킬 수 있습니다.



#### 2. 토픽 확인 및 데이터 송수신

1. **현재 실행 중인 토픽 확인**:

   ```
   rostopic list
   ```

   - 실행 중인 모든 토픽 목록이 출력됩니다.

2. **특정 토픽의 데이터 확인**:

   ```
   rostopic echo /turtle1/cmd_vel
   ```

   - `/turtle1/cmd_vel` 토픽에서 퍼블리싱되는 데이터를 실시간으로 확인.

3. **토픽에 데이터 퍼블리싱**:

   ```
   rostopic pub /turtle1/cmd_vel geometry_msgs/Twist '{linear: {x: 2.0}, angular: {z: 1.0}}'
   ```

   - 거북이가 지정된 속도로 움직입니다.

   

#### 3. rqt_graph를 통한 노드와 토픽 간 관계 확인

1. **rqt_graph 실행**:

   ```
   rqt_graph
   ```

   - 노드와 토픽 간의 관계를 시각적으로 확인할 수 있는 그래프가 표시됩니다.

2. **노드와 토픽 구분**

   ![image](https://github.com/user-attachments/assets/b01573f8-88f8-439f-bf5f-70fa3f84275e)

   - 노드
     - /teleop_turtle : turtle1/cmd_vel 정보를 publish
     - /turtlesim: turtle1/cmd_vel 정보를 subscribe
   - 토픽
     - turtle1/cmd_vel 정보



### 실습: Talker & Listener 

#### 1. 노드와 토픽 실행

1. launch 파일 실행

   ```bash
   roslaunch rospy_tutorials talker_listener.launch
   ```

2. 실행 결과

   - talker가 생성한 내용을 listener가 메시지로 받아서 화면에 출력한다.

   ![image](https://user-images.githubusercontent.com/91526930/234394784-a24bfbb2-8f10-443e-b23d-f5dafda2532e.png)

3. 노드와 토픽 관찰하기

   ![image](https://user-images.githubusercontent.com/91526930/234394161-ca099b10-639c-466d-9162-7fe709a4a39a.png)

#### 2. 관련 코드 확인하기

1. Tutorial 폴더 접근

   ```
   roscd rospy_tutorials
   code .
   ```

2. 폴더 구성 확인

   - `.launch`와 `.py` 로 구성되어 있음.

   ![image](https://user-images.githubusercontent.com/91526930/234396103-730b952f-d540-4871-b962-3101a73b3778.png)

3. `talker_listener.launch`

   - 두 개의 node를 실행하는 것으로 구성됨.
   - 각 node를 초기화하는 구문은 각각의 python 파일에 작성되어 있음.

   ![image](https://user-images.githubusercontent.com/91526930/234396233-154876be-05dc-4bba-b92e-f6e1e1acc233.png)

   

4. `listener.py`

   ![image](https://user-images.githubusercontent.com/91526930/234396748-210f85b3-f6da-42a1-8e1e-434460f27045.png)

   - 'listener'라는 노드를 `init_node`를 통해 초기화한다.

   - 'chatter'라는 String 형태의 msg 정보를 subscribing하고, callback함수를 실행한다. 

   - callback 함수는 data를 받아서, terminal 창에 출력한다.




 5. `talker.py`

    ![image](https://user-images.githubusercontent.com/91526930/234398302-2ef57b3a-b3d7-4d62-966b-13475a1e5971.png)

    - 'chatter'라는 String 형태의 msg 정보를 publishing하는 변수 pub을 선언한다.
    - 'talker'라는 노드를 초기화한다.
    - hello_str에는 String 형태의 정보를 생산 및 할당한다.
    - publish 함수를 통해 hello_str 변수를 publishing 한다.



### 요약

노드와 토픽은 ROS 시스템의 핵심 통신 메커니즘입니다. 이 실습을 통해 노드 실행, 토픽 데이터 송수신, 그리고 노드 간 관계를 시각적으로 확인하는 과정을 익힐 수 있습니다. 이를 기반으로 더 복잡한 시스템을 설계할 수 있습니다.



# Concept and Practice of Nodes and Topics

## What are Nodes and Topics?

### Node
- A **node** is an executable program unit in ROS that performs a specific task (e.g., collecting sensor data, controlling motors).  
- Each node runs independently and communicates with other nodes to exchange data.  

### Topic
- A **topic** is an asynchronous communication channel for data exchange between nodes.  
- Data is exchanged via **Publishers** and **Subscribers**.  
- Example: A camera node publishes image data to the `/camera/image_raw` topic, and another node subscribes to this topic to process the images.  

---

## Practice: TurtleSim

### 1. Run ROS Nodes
- **Run ROS Master**: Start `roscore`, the central service for all ROS communication.  

```bash
roscore
```

2. **Run the Turtlesim Node:**:

   ```
   rosrun turtlesim turtlesim_node
   ```

   - A GUI window opens and a turtle appears.

3. **Run the Turtlesim Control Node**:

   ```
   rosrun turtlesim turtle_teleop_key
   ```

   - You can control the turtle using your keyboard.



#### 2. Check Topics and Send/Receive Data

1. **Check current topics**:

   ```
   rostopic list
   ```

   - Displays all active topics.

2. **Check data of a specific topic**:

   ```
   rostopic echo /turtle1/cmd_vel
   ```

   - Displays the real-time data being published to `/turtle1/cmd_vel`.

3. **Publish data to a topic**:

   ```
   rostopic pub /turtle1/cmd_vel geometry_msgs/Twist '{linear: {x: 2.0}, angular: {z: 1.0}}'
   ```

   - The turtle moves with the specified velocity.

   

#### 3. Visualize Node-Topic Relationships with rqt_graph

1. **Run rqt_graph**:

   ```
   rqt_graph
   ```

   - Displays a graph showing the relationships between nodes and topics.

2. **Topic / Node**

   ![image](https://github.com/user-attachments/assets/b01573f8-88f8-439f-bf5f-70fa3f84275e)

   - **Nodes**
      - `/teleop_turtle`: Publishes turtle1/cmd_vel information.
      - `/turtlesim`: Subscribes to turtle1/cmd_vel information.
   - **Topic**
     - `turtle1/cmd_vel` information



### Practice: Talker & Listener

#### 1. Run Nodes and Topics

1. Run the launch file

   ```bash
   roslaunch rospy_tutorials talker_listener.launch
   ```

2. Result

   - The `talker` node publishes messages, and the `listener` node subscribes to them and prints the messages to the screen.

   ![image](https://user-images.githubusercontent.com/91526930/234394784-a24bfbb2-8f10-443e-b23d-f5dafda2532e.png)

3. 

   ![image](https://user-images.githubusercontent.com/91526930/234394161-ca099b10-639c-466d-9162-7fe709a4a39a.png)

#### 2. Check the Related Code

1. Access the Tutorial folder

   ```
   roscd rospy_tutorials
   code .
   ```

2. The folder contains `.launch` and `.py` files.

   ![image](https://user-images.githubusercontent.com/91526930/234396103-730b952f-d540-4871-b962-3101a73b3778.png)

3. `talker_listener.launch`

   - Configured to run two nodes.
   - Each node is initialized in its respective Python file.

   ![image](https://user-images.githubusercontent.com/91526930/234396233-154876be-05dc-4bba-b92e-f6e1e1acc233.png)

   

4. `listener.py`

   ![image](https://user-images.githubusercontent.com/91526930/234396748-210f85b3-f6da-42a1-8e1e-434460f27045.png)

   - Initializes a node named `listener` using `init_node`.

   - Subscribes to the `chatter` topic (String type).

   - Executes a callback function when messages are received.
  
   - The callback prints the received data to the terminal.




 5. `talker.py`

    ![image](https://user-images.githubusercontent.com/91526930/234398302-2ef57b3a-b3d7-4d62-966b-13475a1e5971.png)

    - Declares a publisher variable `pub` for the `chatter` topic (String type).
    - Initializes a node named `talker`.
    - Generates and assigns a String message to `hello_str`.
    - Publishes `hello_str` to the `chatter` topic.



### Summary

Nodes and topics are the core communication mechanisms in ROS.
Through this practice, you learn how to run nodes, send/receive topic data, and visually confirm relationships between nodes using `rqt_graph`.
These fundamentals allow you to design and expand more complex ROS systems.
