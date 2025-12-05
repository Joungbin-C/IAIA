## Publisher와 Subscriber

#### **Publisher**

- 데이터를 송신하는 노드로, 특정 토픽을 통해 데이터를 퍼블리싱.
- 예: 센서 데이터 전송 노드.

#### **Subscriber**

- 데이터를 수신하는 노드로, 특정 토픽을 구독하여 데이터를 처리.
- 예: 수신된 센서 데이터를 분석하는 노드.

이 두 개념은 ROS에서 노드 간의 비동기 데이터 통신을 구성하며, 노드 간 연결성을 이해하는 데 핵심적입니다.



### 실습: Publisher와 Subscriber 작성

#### 1. Python으로 간단한 퍼블리셔(Publisher) 노드 작성

아래는 문자열 데이터를 퍼블리싱하는 간단한 Python 퍼블리셔 노드 예제입니다:

```
# src/my_package/simple_publisher.py
import rospy
from std_msgs.msg import String

rospy.init_node('simple_publisher')
pub = rospy.Publisher('chatter', String, queue_size=10)
rate = rospy.Rate(1)  # 1 Hz
while not rospy.is_shutdown():
    pub.publish("Hello, ROS!")
    rate.sleep()
```



#### 2. 실행 파일 권한 부여 및 실행

퍼블리셔 노드를 실행하려면 파일에 실행 권한을 부여하고 실행합니다:

```
chmod +x src/my_package/simple_publisher.py
rosrun my_package simple_publisher.py
```



#### 3. 퍼블리싱된 메시지 확인

토픽 `/chatter`에 퍼블리싱된 데이터를 확인합니다:

```
rostopic echo /chatter
```

출력 예시:

```
data: "Hello, ROS!"
```



#### 4. Python으로 간단한 서브스크라이버(Subscriber) 노드 작성

아래는 퍼블리싱된 문자열 데이터를 구독하는 Python 서브스크라이버 노드 예제입니다:

```
# src/my_package/simple_subscriber.py
import rospy
from std_msgs.msg import String

def callback(data):
    rospy.loginfo(f"I heard: {data.data}")

rospy.init_node('simple_subscriber')
sub = rospy.Subscriber('chatter', String, callback)
rospy.spin()
```



#### 5. 서브스크라이버 실행

파일에 실행 권한을 부여한 후 실행합니다:

```
chmod +x src/my_package/simple_subscriber.py
rosrun my_package simple_subscriber.py
```



#### 6. rqt_graph를 통한 통신 관계 확인

퍼블리셔와 서브스크라이버 간의 관계를 시각적으로 확인합니다:

```
rqt_graph
```

- 노드 `simple_publisher`와 `simple_subscriber`가 토픽 `chatter`를 통해 연결된 그래프를 확인할 수 있습니다.



### 실습: C++로 Publisher와 Subscriber 작성

#### 1. C++ 퍼블리셔(Publisher) 노드 작성

아래는 문자열 데이터를 퍼블리싱하는 간단한 C++ 퍼블리셔 노드 예제입니다:

```
// src/my_package/src/simple_publisher.cpp
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "simple_publisher");
    ros::NodeHandle n;

    ros::Publisher pub = n.advertise<std_msgs::String>("chatter", 1000);
    ros::Rate loop_rate(1);

    while (ros::ok())
    {
        std_msgs::String msg;
        std::stringstream ss;
        ss << "Hello, ROS!";
        msg.data = ss.str();

        ROS_INFO("%s", msg.data.c_str());

        pub.publish(msg);
        loop_rate.sleep();
    }

    return 0;
}
```



#### 2. C++ 서브스크라이버(Subscriber) 노드 작성

아래는 퍼블리싱된 문자열 데이터를 구독하는 간단한 C++ 서브스크라이버 노드 예제입니다:

```
// src/my_package/src/simple_subscriber.cpp
#include "ros/ros.h"
#include "std_msgs/String.h"

void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
    ROS_INFO("I heard: [%s]", msg->data.c_str());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "simple_subscriber");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);
    ros::spin();

    return 0;
}
```



#### 3. CMakeLists.txt 수정

`CMakeLists.txt` 파일에 새 노드를 추가합니다:

```
add_executable(simple_publisher src/simple_publisher.cpp)
target_link_libraries(simple_publisher ${catkin_LIBRARIES})

add_executable(simple_subscriber src/simple_subscriber.cpp)
target_link_libraries(simple_subscriber ${catkin_LIBRARIES})
```



#### 4. 워크스페이스 빌드

```
cd ~/catkin_ws
catkin_make
```



#### 5. 노드 실행

```
rosrun my_package simple_publisher
rosrun my_package simple_subscriber
```



#### 6. rqt_graph를 통한 통신 관계 확인

퍼블리셔와 서브스크라이버 간의 관계를 시각적으로 확인합니다:

```
rqt_graph
```





### Python 기반 노드와 C++ 기반 노드의 차이점

#### 1. **빌드 과정**

- **Python**:
  - Python 스크립트는 인터프리터 언어로, 컴파일 과정이 필요 없습니다.
  - 실행 권한(`chmod +x`)만 부여하면 바로 실행할 수 있습니다.
- **C++**:
  - 컴파일된 실행 파일을 생성해야 하므로, `CMakeLists.txt`를 수정하고 워크스페이스를 빌드(`catkin_make`)해야 합니다.



#### 2. **CMakeLists.txt 수정 여부**

- **Python**: CMakeLists.txt 수정이 필요하지 않습니다.
- **C++**: 새로운 노드를 추가할 때마다 `CMakeLists.txt`에 소스 파일과 빌드 타겟을 정의해야 합니다.



#### 3. **package.xml 수정 여부**

- **공통점**:
  - Python과 C++ 모두 사용하는 메시지 타입(`std_msgs` 등)에 대한 의존성을 `package.xml`에 추가해야 합니다.



#### 4. **실행 방식**

- **Python**:

  ```
  rosrun my_package simple_publisher.py
  ```

- **C++**:

  ```
  rosrun my_package simple_publisher
  ```



### 요약

Python 기반 노드는 간단하고 빠르게 실행할 수 있는 반면, C++ 기반 노드는 추가적인 빌드 과정이 필요하지만 성능 면에서 장점이 있습니다. 두 방식을 모두 익혀야 다양한 ROS 프로젝트에서 유연하게 대처할 수 있습니다.