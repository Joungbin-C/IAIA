# Msg 파일

### 개념 설명

ROS의 **Msg 파일**은 노드 간 통신에서 사용하는 데이터 구조를 정의합니다. Msg 파일을 활용하면 다양한 데이터 형식을 표준화하여 퍼블리셔와 서브스크라이버 간의 데이터를 교환할 수 있습니다.



#### **Msg 파일의 주요 특징**

- 노드 간 데이터 송수신에 사용되는 메시지 형식 정의.
- 단순한 데이터 구조에서 복합 데이터 구조까지 지원.
- 메시지 타입은 ROS 표준 라이브러리(`std_msgs`)와 사용자 정의 메시지(Custom Msg)로 나뉩니다.



#### **Msg 파일의 종류**

- **표준 메시지(std_msgs)**:  (예: `String`, `Int32`, `Float64` 등).
  - ROS에서 기본적으로 제공하는 메시지 타입
  - `Bool`, `Byte`, `Int16`, `Float32`, `Int8MultiArray`, `String`, `Time` 등
  - [ROS Wiki - std_msgs](http://wiki.ros.org/std_msgs)
  
- **커스텀 메시지(Custom Msg)**: 사용자가 정의한 메시지 구조로, 프로젝트의 특정 요구에 맞게 설계.



### 커스텀 Msg 파일 생성과 활용

#### 실습 1: 간단한 커스텀 Msg 파일 생성

1. `msg` **디렉토리 생성** 커스텀 메시지를 저장할 디렉토리를 생성합니다:

   ```
   mkdir -p ~/catkin_ws/src/my_package/msg
   ```

2. **Msg 파일 생성** `Person.msg` 파일을 생성하여 아래 내용을 작성합니다:

   ```
   echo -e "string name\nint32 age\nfloat32 height" > ~/catkin_ws/src/my_package/msg/Person.msg
   ```

   이 메시지는 이름, 나이, 키 정보를 담는 구조를 정의합니다.

3. `CMakeLists.txt` **수정** 커스텀 메시지를 빌드 과정에 포함시킵니다:

   ```
   ## 메세지 생성 패키지 추가
   find_package(catkin REQUIRED COMPONENTS
     roscpp
     rospy
     std_msgs
     message_generation
   )
   
   ## 메시지 파일 추가
   add_message_files(
     FILES
     Person.msg
   )
   
   ## 의존성 추가
   generate_messages(
     DEPENDENCIES
     std_msgs
   )
   
   ## 메시지 포함 라이브러리 설정
   catkin_package(
     CATKIN_DEPENDS rospy std_msgs message_runtime
   )
   ```

4. `package.xml` **수정** 메시지 의존성을 추가합니다:

   ```
   <build_depend>message_generation</build_depend>
   <exec_depend>message_runtime</exec_depend>
   ```

5. **워크스페이스 빌드** 새로운 Msg 파일을 빌드하여 자동으로 생성된 헤더 파일을 준비합니다:

   ```
   cd ~/catkin_ws
   catkin_make
   ```

6. **생성된 파일 확인** 빌드 후 `devel/include/my_package`에 `Person.h`헤더 파일의 생성 여부를 확인합니다. 다음 실습 과정에서, 스크립트 작성 시에 해당 헤더 파일을 참조하게 됩니다.



#### 실습 2: 커스텀 Msg 파일 활용 (Python)

1. **퍼블리셔 스크립트 작성**

   - `~catkin_ws/src/my_package/src`위치에서 `custom_publisher.py` 파일 작성:

   ```
   #!/usr/bin/env python3
   #-*- coding:utf-8 -*-
   
   import rospy
   from my_package.msg import Person
   
   rospy.init_node('custom_publisher')
   pub = rospy.Publisher('person_info', Person, queue_size=10)
   rate = rospy.Rate(1)  # 1 Hz
   
   while not rospy.is_shutdown():
       msg = Person()
       msg.name = "John Doe"
       msg.age = 30
       msg.height = 1.75
   
       rospy.loginfo(f"Publishing: {msg}")
       pub.publish(msg)
       rate.sleep()
   ```

2. **서브스크라이버 스크립트 작성**

   - `~catkin_ws/src/my_package/src`위치에서 `custom_subscriber.py` 파일 작성:

   ```
   #!/usr/bin/env python3
   #-*- coding:utf-8 -*-
   
   import rospy
   from my_package.msg import Person
   
   def callback(data):
       rospy.loginfo(f"Received: Name={data.name}, Age={data.age}, Height={data.height}")
   
   rospy.init_node('custom_subscriber')
   sub = rospy.Subscriber('person_info', Person, callback)
   rospy.spin()
   ```

3. **노드 실행** 퍼블리셔와 서브스크라이버를 실행하여 커스텀 메시지를 활용한 데이터를 주고받습니다:

   ```
   chmod +x ~/catkin_ws/src/my_package/src/custom_publisher.py
   chmod +x ~/catkin_ws/src/my_package/src/custom_subscriber.py
   roscore
   rosrun my_package custom_publisher.py
   rosrun my_package custom_subscriber.py
   ```

 - 오류발생 (my_package.msg) 찾지 못할경우
   ```
   source ~/catkin_ws/devel/setup.bash 
   ```
    실행 후 재실행

#### 실습 3: 커스텀 Msg 파일 활용 (C++)

1. **퍼블리셔 스크립트 작성**

   -  `~catkin_ws/src/my_package/src`위치에서 `custom_publisher.cpp` 파일 작성:

   ```
   #include "ros/ros.h"
   #include "my_package/Person.h"
   
   int main(int argc, char **argv)
   {
       ros::init(argc, argv, "custom_publisher");
       ros::NodeHandle n;
   
       ros::Publisher pub = n.advertise<my_package::Person>("person_info", 1000);
       ros::Rate loop_rate(1);
   
       while (ros::ok())
       {
           my_package::Person msg;
           msg.name = "John Doe";
           msg.age = 30;
           msg.height = 1.75;
   
           ROS_INFO("Publishing: Name=%s, Age=%d, Height=%.2f", msg.name.c_str(), msg.age, msg.height);
           pub.publish(msg);
   
           loop_rate.sleep();
       }
   
       return 0;
   }
   ```

2. **C++ 서브스크라이버에서 사용** `custom_subscriber.cpp` 파일 작성:

   ```
   // src/my_package/src/custom_subscriber.cpp
   #include "ros/ros.h"
   #include "my_package/Person.h"
   
   void personCallback(const my_package::Person::ConstPtr& msg)
   {
       ROS_INFO("Received: Name=%s, Age=%d, Height=%.2f", msg->name.c_str(), msg->age, msg->height);
   }
   
   int main(int argc, char **argv)
   {
       ros::init(argc, argv, "custom_subscriber");
       ros::NodeHandle n;
   
       ros::Subscriber sub = n.subscribe("person_info", 1000, personCallback);
       ros::spin();
   
       return 0;
   }
   ```

3. **CMakeLists.txt 수정** `custom_publisher`와 `custom_subscriber` 노드를 빌드에 추가:

   ```
   add_executable(custom_publisher src/custom_publisher.cpp)
   target_link_libraries(custom_publisher ${catkin_LIBRARIES})
   
   add_executable(custom_subscriber src/custom_subscriber.cpp)
   target_link_libraries(custom_subscriber ${catkin_LIBRARIES})
   ```

4. **워크스페이스 빌드 및 실행**

   ```
   cd ~/catkin_ws
   catkin_make
   
   rosrun my_package custom_publisher
   rosrun my_package custom_subscriber
   ```



### Msg 파일 빌드 과정 시각화

1. **Msg 파일 생성**: `msg/Person.msg` 작성.
2. **빌드 시스템에 통합**: `CMakeLists.txt` 및 `package.xml` 수정.
3. **워크스페이스 빌드**: `catkin_make` 실행.
4. **자동 생성된 헤더 파일**: `devel/include/my_package`에 헤더 파일 생성.
5. **퍼블리셔와 서브스크라이버에서 헤더 파일 사용**: 생성된 헤더 파일을 참조.



### 요약

Msg 파일은 ROS에서 데이터 구조를 표준화하여 노드 간 통신을 가능하게 합니다. 표준 메시지 외에도 커스텀 메시지를 활용하여 프로젝트에 맞는 데이터를 정의하고 효율적으로 교환할 수 있습니다. 실습을 통해 Msg 파일의 생성 및 활용 과정을 익히고, ROS 시스템의 유연성을 경험해 보세요.







# Msg Files

## Concept

ROS **Msg files** define the data structures used for communication between nodes.  
By using Msg files, various data formats can be standardized, enabling smooth data exchange between publishers and subscribers.  

---

## Key Features of Msg Files
- Define message formats used for node-to-node communication.  
- Support from simple to complex data structures.  
- Message types are divided into:
  - **Standard Messages (std_msgs)**  
  - **Custom Messages (Custom Msgs)**  

---

## Types of Msg Files

### Standard Messages (std_msgs)
Examples: **String, Int32, Float64, Bool, Byte, Int16, Float32, Int8MultiArray, Time, etc.**  
Provided by default in ROS.  
[ROS Wiki - std_msgs](http://wiki.ros.org/std_msgs)

### Custom Messages (Custom Msgs)
User-defined message structures designed for the specific requirements of a project.  

---

## Creating and Using Custom Msg Files

### Practice 1: Create a Simple Custom Msg File

1. **Create msg directory**  
Create a directory to store custom messages:

```bash
mkdir -p ~/catkin_ws/src/my_package/msg
 ```

2. **Create Msg File** : Create a file named `Person.msg` with the following content:

   ```
   echo -e "string name\nint32 age\nfloat32 height" > ~/catkin_ws/src/my_package/msg/Person.msg
   ```

   This defines a structure containing name, age, and height.

3. Modify `CMakeLists.txt`
   - Include custom messages in the build process:

   ```
    ## Add message generation dependency
    find_package(catkin REQUIRED COMPONENTS
      roscpp
      rospy
      std_msgs
      message_generation
    )
    
    ## Add message files
    add_message_files(
      FILES
      Person.msg
    )
    
    ## Add dependencies
    generate_messages(
      DEPENDENCIES
      std_msgs
    )
    
    ## Set message runtime
    catkin_package(
      CATKIN_DEPENDS rospy std_msgs message_runtime
    )
   ```

5. Modify package.xml
   - Add message dependencies:

   ```
   <build_depend>message_generation</build_depend>
   <exec_depend>message_runtime</exec_depend>
   ```

7. Build Workspace

   ```
   cd ~/catkin_ws
   catkin_make
   ```

9. **Check Generated Files** :
    - After building, verify the header file in:
      
    ```
    devel/include/my_package/Person.h
    ```


#### Practice 2: Using Custom Msg (Python)

1. **Publisher Script**

   - `custom_publisher.py` at `~catkin_ws/src/my_package/src`:

   ```
   #!/usr/bin/env python3
   #-*- coding:utf-8 -*-
   
   import rospy
   from my_package.msg import Person
   
   rospy.init_node('custom_publisher')
   pub = rospy.Publisher('person_info', Person, queue_size=10)
   rate = rospy.Rate(1)  # 1 Hz
   
   while not rospy.is_shutdown():
       msg = Person()
       msg.name = "John Doe"
       msg.age = 30
       msg.height = 1.75
   
       rospy.loginfo(f"Publishing: {msg}")
       pub.publish(msg)
       rate.sleep()
   ```

2. **Subscriber Script**

   - `custom_subscriber.py` at `~catkin_ws/src/my_package/src`
     
   ```
   #!/usr/bin/env python3
   #-*- coding:utf-8 -*-
   
   import rospy
   from my_package.msg import Person
   
   def callback(data):
       rospy.loginfo(f"Received: Name={data.name}, Age={data.age}, Height={data.height}")
   
   rospy.init_node('custom_subscriber')
   sub = rospy.Subscriber('person_info', Person, callback)
   rospy.spin()
   ```

3. **Run Nodes**

   ```
   chmod +x ~/catkin_ws/src/my_package/src/custom_publisher.py
   chmod +x ~/catkin_ws/src/my_package/src/custom_subscriber.py
   roscore
   rosrun my_package custom_publisher.py
   rosrun my_package custom_subscriber.py
   ```

 ⚠️ If you see an error (my_package.msg not found), run:
 
   ```
   source ~/catkin_ws/devel/setup.bash 
   ```


#### Practuce 3: Using Custom Msg (C++)

1. **Publisher Script**

   - `custom_publisher.cpp` at `~catkin_ws/src/my_package/src`

   ```
   #include "ros/ros.h"
   #include "my_package/Person.h"
   
   int main(int argc, char **argv)
   {
       ros::init(argc, argv, "custom_publisher");
       ros::NodeHandle n;
   
       ros::Publisher pub = n.advertise<my_package::Person>("person_info", 1000);
       ros::Rate loop_rate(1);
   
       while (ros::ok())
       {
           my_package::Person msg;
           msg.name = "John Doe";
           msg.age = 30;
           msg.height = 1.75;
   
           ROS_INFO("Publishing: Name=%s, Age=%d, Height=%.2f", msg.name.c_str(), msg.age, msg.height);
           pub.publish(msg);
   
           loop_rate.sleep();
       }
   
       return 0;
   }
   ```

2. **C++ Subscriber Script
   - ** `custom_subscriber.cpp` 

   ```
   // src/my_package/src/custom_subscriber.cpp
   #include "ros/ros.h"
   #include "my_package/Person.h"
   
   void personCallback(const my_package::Person::ConstPtr& msg)
   {
       ROS_INFO("Received: Name=%s, Age=%d, Height=%.2f", msg->name.c_str(), msg->age, msg->height);
   }
   
   int main(int argc, char **argv)
   {
       ros::init(argc, argv, "custom_subscriber");
       ros::NodeHandle n;
   
       ros::Subscriber sub = n.subscribe("person_info", 1000, personCallback);
       ros::spin();
   
       return 0;
   }
   ```

4. Modify CMakeLists.txt to add executables:

   ```
   add_executable(custom_publisher src/custom_publisher.cpp)
   target_link_libraries(custom_publisher ${catkin_LIBRARIES})
   
   add_executable(custom_subscriber src/custom_subscriber.cpp)
   target_link_libraries(custom_subscriber ${catkin_LIBRARIES})
   ```

5. Build and Run:

   ```
   cd ~/catkin_ws
   catkin_make
   
   rosrun my_package custom_publisher
   rosrun my_package custom_subscriber
   ```



### Visualization of Msg File Build Process

1. **Create Msg File**: Write `msg/Person.msg` 
2. **Integrate into Build System**: Modify `CMakeLists.txt` and `package.xml`
3. **Build Workspace**: Run `catkin_make`
4. **Auto-generated Header File**: Located in `devel/include/my_package`
5. **Use in Publisher/Subscriber**: Import and reference generated headers in your code.



### Summary

Msg files in ROS standardize data structures for communication between nodes.
Beyond standard messages, custom messages allow defining project-specific data formats for efficient exchange.
Through this practice, you learn how to create and use Msg files, enhancing the flexibility and scalability of the ROS system.
