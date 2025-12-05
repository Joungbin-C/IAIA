# ROS Build System - rospy





## 패키지 빌드 과정

1. 패키지 생성
2. 패키지 설정 파일(package.xml) 수정
3. 빌드 설정 파일(CMakeList.txt) 수정
4. 메시지 파일 작성
5. 소스 코드 작성
6. 빌드 전 처리
7. 노드 실행
8. Launch 실행



## 1. package 생성

``` bash
[위치] ~/catkin_ws/src
catkin_create_pkg [패키지 이름] [의존성 패키지]
catkin_create_pkg my_tutorial std_msgs rospy roscpp
```

<img src="https://user-images.githubusercontent.com/91526930/235348013-d1cbeda0-7d0c-461d-867f-8d4a966dd032.png" alt="image" style="zoom:50%;" />

- `/include`: 헤더 파일
- `/src`: 코드 소스 파일
- `CMakeList.txt`: 빌드 설정 파일
- `package.xml`: 패키지 설정 파일

추후, 사용자의 필요에 따라 `/launch`, `/msg` 등의 폴더를 추가적으로 생성할 수 있음.



## 2. 패키지 설정 파일(package.xml) 수정

<img src="https://user-images.githubusercontent.com/91526930/235348353-581b5ef9-56ce-4f6d-9849-07f8a71d92b4.png" alt="image" style="zoom:67%;" />

- 기본 구조

  - `<?xml>` : 문서의 문법을 정의함. xml의 버전을 나타냄.
  - `<package>`: 해당 태그의 내부는 ROS 패키지의 설정관 관련한 내용이 담겨 있음.

- 패키지 정보

  - `<name>`: 패키지의 이름. 패키지 생성시 입력한 이름이 적용되며, 사용자의 임의 변경이 가능함.
  - `<version>`: 패키지 버전으로, 자유로운 지정이 가능함.
  - `<description>`: 패키지에 대한 설명. 2-3문장으로 입력함.
  - `<maintainer>`: 패키지 관리자의 이름과 메일 주소(태그의 옵션 email을 이용)를 입력.

- 의존 패키지(Dependency)

  - `<buildtool_depend>`: 빌드 시스템의 의존성이며, Catkin 빌드 시스템을 이용한다면 `catkin`을 입력.
  - `<build_depend>`: 패키지를 빌드할 때 의존하는 패키지 이름을 입력.
  - `<build_export_depend>`: 패키지를 내보내기 할 때 의존하는 패키지 이름을 입력.
  - `<exec_depend>`: 패키지를 실행할 때 의존하는 패키지 이름을 입력.

- 메타 패키지(Metapackage)

  - `<export>`: ROS에서 명시하지 않은 태그명을 사용할 때 주로 사용.
  - `<metapackage>`: export 태그 안에서 사용하는 공식적인 태그 중 하나로, 현재 패키지가 메타패키지일 경우 선언.

  

패키지 생성 당시, `std_msgs`, `rospy`, `roscpp` 의존성 패키지를 입력하였으므로, 자동으로 `<build_depend>`, `<build_export_depend>`, `<exec_depend>`에 입력되어 있음. 필요에 따라, 해당 태그 내에 필요한 의존성 패키지를 추가할 수 있음.



## 3. 빌드 설정 파일(CMakeList.txt) 수정

아래의 내용은 `CMakeList.txt` 의 내용을 정리하기 위함이므로, 필요에 따라 수정하길 바람. (더 자세한 내용은 reference 자료에 있음)

```cmake
# 운영체제에 설치된 cmake의 최소 요구 버전
cmake_minimum_required(VERSION 3.0.2)

# 패키지의 이름. package.xml에서 입력한 패키지 이름 그대로 사용
project(my_tutorial)

# cakin 빌드 시 요구되는 패키지. 사용자가 생성한 패키지가 의존하는 다른 패키지를 먼저 설치하는 옵션
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

# 파이썬을 이용하기 위해 rospy를 사용할 때 설정하는 옵션. 파이썬 설치 프로세스인 setup.py를 부르는 역할
catkin_python_setup()

# 메시지 파일 추가
# FILES: 현재 패키지 폴더의 msg 폴더 안의 .msg 파일들을 참조해 헤더 파일(.h)를 자동으로 생성
# 사용자가 메세지를 추가한 경우, msg 폴더를 만든 뒤 그 안에 있는 메시지 파일을 입력함.
# 아래의 예시는 /msg 내부에 Message1.msg과 Message2.msg라는 사용자가 정의한 메세지가 존재하며, 해당 메세지를 패키지에서 사용하겠다는 것을 의미함.
add_message_files(
  FILES
  Message1.msg
  Message2.msg
)

# add_service_files, add_action_files 마찬가지(reference 참고)

# 의존하는 메시지 설정
# DEPENDENCIES: 아래에 해당하는 메시지 패키지를 사용한다는 의미
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

# cakin build 옵션
# INCLUDE: 뒤에 설정한 패키지 내부 폴더인 include의 헤더 파일을 사용함
# LIBRARIES: 뒤에 설정한 패키지의 라이브러리를 사용함
# CATKIN_DEPENDS: 의존하는 패키지 지정
# DEPENDS: 시스템 의존 패키지
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES my_tutorial
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

# include 폴더 지정
include_directories(
  ${catkin_INCLUDE_DIRS} # 각 패키지 내의 include 폴더를 의미. 이 안의 헤더파일을 이용할 것. 
  # 사용자가 추가할 때는 이 밑의 공간 이용
)

# 빌드 후 생성할 라이브러리. C++을 사용할 경우!
add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}/test_pkg.cpp
)

# 해당 라이브러리 및 실행파일을 빌드하기 전, 생성해야 할 의존성이 있는 메시지와 dynamic reconfigure이 있다면 우선으로 수행하도록 함
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 빌드 후 생성할 실행파일에 대한 옵션 지정
## `__실행 파일 이름__` `__참조할 파일__` 순서대로 기재
## 복수 개의 참조 .cpp 파일이 있을 경우 한 괄호 뒤에 연속적으로 기재
## 생성할 실행파일이 2개 이상일 경우 add_executable 항목을 추가함
add_executable(${PROJECT_NAME}_node src/test_pkg_node.cpp)

# 지정 실행 파일을 생성하기 전, 링크해야 하는 라이브러리와 실행파일을 링크함
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)
```





## 4. 메시지 파일 작성

- `msg` 폴더 및 `tutorial_msg.msg` 파일 생성

  ![image](https://user-images.githubusercontent.com/91526930/235349822-b71067ee-b147-4d00-a547-aaaf73d0315d.png)

- `tutorial_msg.msg` 파일 내부에 다음과 같이 입력 후 저장.

```
time stamp
int32 data
```



## 5. 소스 코드 작성

### python 버전

- `my_tutorial/src` 폴더 내부에 `talker.py`를 생성하여, 다음 내용을 입력.

  ```python
  #!/usr/bin/env python3	# 파이썬을 사용하기 위해 꼭 필요함.
  #-*- coding:utf-8 -*-	# 한글 주석을 작성하기 위해 필요함.
  
  import rospy								# ROS 라이브러리
  from my_tutorial.msg import tutorial_msg	# my_tutorial 패키지 내부에 정의된 msg 중 tutorial_msg를 참고하겠다는 의미.
  
  def main():
      # 노드 초기화
      rospy.init_node('talker', anonymous=True)  # 노드 이름: talker
      
      # Publisher 초기화 및 변수 선언
      pub = rospy.Publisher('chatter', tutorial_msg, queue_size=10) # topic: chatter, msg_type: tutorial_msg
      
      # 변수 선언
      rate = rospy.Rate(10) 	# 10 Hz 주기 설정
      msg = tutorial_msg()	# 메시지 변수
  
      count = 0
      
      # 프로그램 반복 실행
      while not rospy.is_shutdown():		# 프로그램 중단(shut down)되기 전까지 계속 실행
          msg.stamp = rospy.Time.now()	# 현재 시간을 (메시지 변수).stamp에 저장
          msg.data = count				# count 값을 (메시지 변수).data에 저장
  
          # 터미널에 출력
          rospy.loginfo("send time(sec) = %d", msg.stamp.secs)
          rospy.loginfo("send msg = %d", msg.data)
          
          # Publisher 변수를 통해 msg를 publish 함. 
          pub.publish(msg)
          
          rate.sleep()	# 사전에 정의된 주기(Hz)만큼 일시중지 (주기적으로 반복동작하기 위함.)
  
          count += 1
  
  if __name__ == '__main__':
      try:
          main()
      except rospy.ROSInterruptException:
          pass
  ```



- `my_tutorial/src` 폴더 내부에 `listener.py`를 생성하여, 다음 내용을 입력.

```python
#!/usr/bin/env python3
#-*- coding:utf-8 -*- 

import rospy
from my_tutorial.msg import tutorial_msg

# Publisher가 생성한 topic을 Subscribing 하면서 callback 함수 실행
def callback(data):		# topic 정보를 data 변수로 읽어들임.
    # 터미널에 출력
    rospy.loginfo("recieve time(sec) = %d", data.stamp.secs)
    rospy.loginfo("recieve msg = %d", data.data)
    
def main():
    # 노드 초기화
    rospy.init_node('listener', anonymous=True)	# 노드 이름: listener

    # Sbusciber 초기화 및 callback 함수 지정
    rospy.Subscriber("chatter", tutorial_msg, callback)	# topic: chatter, msg_type: tutorial_msg

    # 무한 루프 / 노드가 callback 이외의 어떤 일도 하지 않는다면 spin()함수 사용.
    rospy.spin()

if __name__ == '__main__':
    main()
```



### C++ 버전 

- reference: [ROS wiki - publish & subscribe](http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29)

- `my_tutorial/src` 폴더 내부에 `talker_cpp.cpp`를 생성하여, 다음 내용을 입력.

```cpp
#include "ros/ros.h"					// ROS 기본 헤더 파일
#include "my_tutorial/tutorial_msg.h"	// 메시지 파일의 헤더. 빌드 후 자동 생성됨.

int main(int argc, char **argv){
  // 노드 초기화
  ros::init(argc, argv, "talker");     // 노드 이름: talker
  ros::NodeHandle nh;                  // ROS 시스템과 통신을 위한 노드 핸들
  
  // Publisher 초기화 및 변수 선언 (노드 핸들을 이용하여 publisher 초기화)
  // my_tutorial 패키지에 정의된 tutorial_msg 타입의 '/chatter' 토픽 생성 (queue 사이즈 = 100개)
  // 만약, 표준 메시지를 사용한다면 (예) <std_msgs::String>
  ros::Publisher pub_chatter = nh.advertise<my_tutorial::tutorial_msg>("/chatter", 100); 
  
  // 변수 초기화  
  ros::Rate loop_rate(10);			// 주기를 10Hz로 설정
  my_tutorial::tutorial_msg msg;	// 메시지 변수 선언
  
  int count = 0;

  while (ros::ok()){			// 종료 전까지 반복 수행
    msg.stamp = ros::Time::now();	// 메시지 변수 내 stamp에 현 시간 입력
    msg.data  = count;				// 메시지 변수 내 data에 count 입력

    // 터미널에 출력
    ROS_INFO("send time(sec) = %d", msg.stamp.sec);
    ROS_INFO("send msg = %d", msg.data);

    // publisher 변수를 통해 msg를 publish 함
    pub_chatter.publish(msg);

    loop_rate.sleep();		// 사전에 정의된 주기만큼 일시정지(sleep)

    ++count;
  }

  return 0;
}
```



- `my_tutorial/src` 폴더 내부에 `listener_cpp.cpp`를 생성하여, 다음 내용을 입력.

```cpp
#include "ros/ros.h"
#include "my_tutorial/tutorial_msg.h" 

// Publisher가 생성한 topic을 Subscribing 하면서 callback 함수 실행
void chatterCallback(const my_tutorial::tutorial_msg::ConstPtr& msg){ // 토픽 정보를 msg 포인터 파라미터로 읽어드림.
  // 터미널 출력
  ROS_INFO("recieve time(sec) = %d", msg->stamp.sec);
  ROS_INFO("recieve msg = %d", msg->data);
}

int main(int argc, char **argv){
  // 노드 초기화
  ros::init(argc, argv, "listener");	// 노드 이름: listener
  ros::NodeHandle nh;					// ROS 시스템과 통신을 위한 노드 핸들
    
  // Subscriber 초기화 및 변수 선언, callback 함수 지정
  ros::Subscriber sub = nh.subscribe("/chatter", 100, chatterCallback);	// 토픽 이름: '/chatter', queue size = 100

  // 무한 루프 / 노드가 callback 이외의 어떤 일도 하지 않는다면 spin()함수 사용.
  ros::spin();

  return 0;
}
```





## 6. 빌드 전 처리

### python 버전

- `package.xml` 파일 내부에 다음 부분을 추가.

  ```xml
  <build_depend>message_generation</build_depend>
  <exec_depend>message_runtime</exec_depend>
  ```

- `CMakeList.txt` 파일을 열어서, 다음 내용에 해당하는 부분을 수정할 것. 

  ```cmake
  cmake_minimum_required(VERSION 3.0.2)
  project(my_tutorial)
  
  # message_generation 추가
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
  )
  
  # tutorial_msg.msg 추가
  add_message_files(
    FILES
    tutorial_msg.msg
  )
  
  # 주석 풀기 
  generate_messages(
    DEPENDENCIES
    std_msgs
  )
  
  # LIBRARIES, CATKIN_DEPENDS 주석 풀기
  # message_runtime 추가
  catkin_package(
  #  INCLUDE_DIRS include
   LIBRARIES my_tutorial
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  #  DEPENDS system_lib
  )
  
  # 주석 풀기
  # 스크립트 파일 목록 추가 (talker.py, listener.py)
  catkin_install_python(PROGRAMS
    # scripts/my_python_script
    src/talker.py
    src/listener.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  ```



- 패키지 빌드

  ```bash
  catkin_make
  ```

  

- 파이썬 스크립트에 대해 실행권한 허용.

  ```bash
  [위치] ~/catkin_make/src/my_tutorial/src
  chmod +x talker.py
  chmod +x listener.py
  ```



### C++ 버전

- `package.xml` 파일 내부에 다음 부분을 추가.

  ```xml
  <build_depend>message_generation</build_depend>
  <exec_depend>message_runtime</exec_depend>
  ```
  
  

- `CMakeList.txt` 파일의 내용을 다음과 같이 수정.

  ```cmake
  # message_generation 추가
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
  )
  
  # tutorial_msg.msg 추가
  add_message_files(
    FILES
    tutorial_msg.msg
  )
  
  # 주석 풀기 
  generate_messages(
    DEPENDENCIES
    std_msgs
  )
  
  # CATKIN_DEPENDS 주석 풀기
  # message_runtime 추가
  catkin_package(
  #  INCLUDE_DIRS include
  # LIBRARIES my_tutorial
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
  #  DEPENDS system_lib
  )
  
  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
  )
  
  # add_executable 추가
  # src폴더의 talker_cpp.cpp를 talker_cpp라고 실행한다는 의미.
  add_executable(talker_cpp src/talker_cpp.cpp)
  target_link_libraries(talker_cpp
    ${catkin_LIBRARIES}
  )
  add_dependencies(talker_cpp
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )
  add_executable(listener_cpp src/listener_cpp.cpp)
  target_link_libraries(listener_cpp
    ${catkin_LIBRARIES}
  )
  add_dependencies(listener_cpp
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )
  ```

  

- 패키지 빌드

  ```bash
  catkin_make
  ```

  - 빌드 시 에러가 난다면, 에러 메시지를 읽을 것.
    에러 메시지를 추적해보면, 오타가 많은 부분을 차지함.



## 7. 노드 실행

### 파이썬 버전

```bash
roscore								# terminal 1
rosrun my_tutorial listener.py 		# terminal 2
rosrun my_tutorial talker.py		# terminal 3
rqt_graph							# terminal 4 -> 노드와 토픽을 시각화하여 확인
```

### C++ 버전

```bash
roscore								# terminal 1
rosrun my_tutorial listener_cpp 	# terminal 2
rosrun my_tutorial talker_cpp		# terminal 3
rqt_graph							# terminal 4 -> 노드와 토픽을 시각화하여 확인
```



### 파이썬 & C++ 

```bash
roscore								# terminal 1
rosrun my_tutorial listener.py 		# terminal 2
rosrun my_tutorial talker_cpp		# terminal 3
rosrun my_tutorial listener_cpp 	# terminal 4
rqt_graph							# terminal 5 -> 노드와 토픽을 시각화하여 확인
```



## 8. Launch 실행

- `/launch` 폴더에 `talker_listener.launch` 파일 생성

  ```xml
  <launch>
    <!-- python 버전 -->
    <node name="listener" pkg="my_tutorial" type="listener.py" output="screen"/>
    <node name="talker" pkg="my_tutorial" type="talker.py" output="screen"/>
    
    <!-- c++ 버전 -->
    <node name="listener_cpp" pkg="my_tutorial" type="listener_cpp" output="screen"/>
    <node name="talker_cpp" pkg="my_tutorial" type="talker_cpp" output="screen"/>
  </launch>
  ```

  
