
## ROS Package 설치

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





## ROS Package 복사

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

