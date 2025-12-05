# Work Space와 Build



## 1. Work Space 생성

* ROS에서 워크스페이스는 프로젝트 파일, 페키지 등을 관리하는 기본 작업 디렉토리

### 1.1. 디렉토리 생성

```bash
mkdir -p ~/catkin_ws/src
```

* `catkin_ws`는 워크스페이스 이름이며, 원하는 이름으로 변경 가능



### 1.2. 워크스페이스 초기화

```bash
cd ~/catkin_ws
catkin_make
```

* `catkin_make`를 실행하면 `build/`, `devel/` 디렉토리가 생성됨
* `devel/`은 빌드된 파일이 저장되는 경로



### 1.3. 환경 설정

```bash
source devel/setup.bash
```

* ROS 환경 변수를 활성화하여 워크스페이스를 ROS에서 인식하게 함.

* 매번 터미널을 열 때마다 실행해야 하므로 `~/.bashrc`에 추가하는 것이 편리함.

  ```bash
  echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
  source ~/.bashrc
  ```



### 1.4. 워크스페이스 확인

```bash
echo $ROS_PACKAGE_PATH
```

* ROS가 현재 워크스페이스를 인식하고 있는지 확인





## 2. ROS Build System

- **Catkin**은 ROS의 빌드 시스템이다.
- **CMake(Cross Platform Make)**를 기본적으로 이용.
- `CMakeList.txt`라는 파일에 빌드 환경을 기술해야 함.

정리: CMake를 이용하여, ROS 환경에 알맞게 Catkin 빌드 시스템을 만들었으며, ROS 관련 빌드, 패키지 관리, 패키지 간 의존성 등을 편리하게 사용할 수 있게 되었다. 

**reference**: [ROS의 빌드시스템](https://velog.io/@717lumos/ROS-%ED%8C%A8%ED%82%A4%EC%A7%80-%EB%B9%8C%EB%93%9C%EC%99%80-%EB%85%B8%EB%93%9C-%EC%9E%91%EC%84%B1)





### 2.1. 패키지 빌드 과정

1. 패키지 생성
2. 패키지 설정 파일(`package.xml`) 수정
3. 빌드 설정 파일(`CMakeList.txt`) 수정
4. 메시지 파일 작성
5. 소스 코드 작성
6. 빌드 전 처리
7. 노드 실행



### 2.2. 빌드 해야하는 상황

* 새로운 패키지를 추가했을 때
  * Work Space에 새로운 ROS 패키지를 생성, 추가한 경우



* 코드를 수정했을 때
  * 패키지 내부에 Node, Library, Message, Service 등의 코드를 수정한 경우



* 의존성을 추가하거나 변경했을 때
  * `package.xml `혹은 `CMakeLists.txt` 파일에 새로운 의존성(dependence) 추가 혹은 기존 의존성 제거/변경한 경우



* 빌드 환경을 변경했을 때
  * 새로운 라이브러리나 툴을 설치한 경우
  * ROS 버전을 업데이트한 경우



* 새로운 메세지/서비스 타입을 정의했을 때
  * 새로운 메세지(`.msg`)나 서비스(`.srv`) 파일을 수정한 경우



#### Tip: 빌드가 필요 없는 경우

* ROS 파라미터나 `launch` 파일만 수정한 경우



## 3. 기본 명령어

### 3.1. 파일 및 경로 관련 명령어

* 현재 ROS 패키지 경로 이동
  * 특정 패키지의 루트 디렉토리로 바로 이동함

```bash
roscd [package_name]
```



* 패키지 파일 검색
  * 지정한 패키지의 설치 경로를 반환

```bash
rospack find [package_name]
```



### 3.2. 실행 관련 명령어

* 노드 실행
  * 특정 패키지의 노드를 실행

```bash
rosrun [package_name] [node_name]
```



* 런치 파일 실행
  * 런치 파일을 통해 여러 노드를 동시에 실행

```bash
roslaunch [package_name] [launch_file]
```



### 3.3. 상태 확인 명령어

* 노드 정보 확인

```bash
rosnode list
```

* 토픽 정보 확인

```bash
rostopic list
```

* 토픽 메시지 확인
  * 특정 토픽의 메시지 내용을 실시간으로 출력

```bash
rostopic echo [topic_name]
```



### 3.4. 디버깅 명령어

* rqt 도구 실행
  * GUI 기반의 ROS 디버깅 및 모니터링 도구 실행

```bash
rqt
```

* 토픽 그래프 확인
  * 노드와 토픽 간의 관계를 그래프로 시각화

```bash
rqt_graph
```





## 4. 노드와 토픽 개념

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





## 5. ROS Package

- ROS에서 공식적으로 제공하는 pacakge는 보통 다음과 같은 명령어를 통해 설치됨.

  ```bash
  sudo apt-get install ros-noetic-[pacakge name]
  ```

- 위 방식으로 설치된 package는 `roscd` 명령어를 통해 폴더에 접근할 수 있음. 

  ```bash
  roscd [package name]
  ```

- 경로에서 확인할 수 있듯이 package를 설치하면 `/opt/ros/noetic/share` 내부에는 알게 모르게 설치된 수많은 package들이 존재함.
  `vs code`를 통해 프로그램 코드를 모두 확인할 수는 있지만, 수정할 수 있는 권한이 없음.

![image](https://user-images.githubusercontent.com/91526930/235362934-a74b67f4-0026-4bf7-96af-aaeec117a5f3.png)





## 6. 외부 패키지 설치 방법

- ROS에서 공식적으로 만들어서 제공하는 package는 아니지만, 특정 회사나 개인이 만들어 package를 github를 통해 배포하는 경우가 있음.

- ROS 인증(?) 받은 package의 경우, 아래와 같이 설치할 수 있음.

  ```bash
  sudo apt-get install ros-noetic-[pacakge name]
  ```

- 그러나, 해당 package를 직접 수정하여 build하고자 한다면 다음과 같이 `~/catkin_ws/src` 내부에 패키지를 복사하면 가능함.
  (ROS 인증이 되지 않은 package이더라도, 유용한 pacakge로 판단된다면, 복사하여 필요한 작업을 진행하면 됨.)

  ```bash
  [위치] ~/catkin/src
  git clone https://github.com/[USERNAME]/[REPOSITORY_NAME].git
  git clone -b [branch name] https://github.com/[USERNAME]/[REPOSITORY_NAME].git # 특정 branch를 복사해야 하는 경우
  ```

- 패키지 빌드

  ```bash
  [위치] ~/catkin
  catkin_make
  ```