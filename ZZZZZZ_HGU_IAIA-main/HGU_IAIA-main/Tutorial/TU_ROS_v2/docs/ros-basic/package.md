# 패키지의 개념과 생성

### 패키지란 무엇인가?

ROS에서 **패키지(Package)**는 소프트웨어 모듈의 기본 단위로, ROS 시스템에서 다음과 같은 기능적 역할을 수행합니다:

- **데이터 통합**: 노드, 메시지, 서비스 파일을 통해 다양한 데이터 흐름을 효율적으로 관리.
- **시스템 모듈화**: 복잡한 로봇 시스템을 독립적인 소프트웨어 모듈로 나누어 설계 및 유지보수 용이.
- **확장성 지원**: 기존 패키지와 새로운 패키지를 결합하여 로봇 기능 확장.

패키지에는 다음과 같은 요소가 포함됩니다:

1. **노드(Node)**: ROS에서 실행 가능한 프로그램.
2. **메시지 정의 파일(Message Definition Files)**: 노드 간 데이터 구조를 정의.
3. **서비스 정의 파일(Service Definition Files)**: 요청-응답 데이터 구조 정의.
4. **실행 파일**: 컴파일된 실행 가능한 바이너리 또는 스크립트 파일.

패키지는 ROS 시스템에서 독립적으로 관리될 수 있으며, 재사용 가능하도록 설계됩니다.



### 패키지 구조

패키지는 보통 다음과 같은 디렉토리와 파일로 구성됩니다:

- `**CMakeLists.txt**`: 빌드 시스템 설정 파일로, 패키지의 소스 코드와 의존성을 정의.
- `**package.xml**`: 패키지의 메타데이터와 의존성을 관리하는 XML 파일.
- `**src/**`: 노드의 소스 코드가 저장되는 디렉토리. Python 또는 C++로 작성된 파일이 위치.
- `**include/**`: C++ 프로젝트에서 사용되는 헤더 파일 디렉토리.
- `**scripts/**`: 실행 가능한 Python 스크립트가 저장되는 디렉토리.
- `**msg/**`: 사용자 정의 메시지 파일이 저장되는 디렉토리.
- `**srv/**`: 사용자 정의 서비스 파일이 저장되는 디렉토리.
- `**action/**`: 액션 파일이 저장되는 디렉토리(선택적).



### 패키지 생성 실습

#### 1. 패키지 생성

워크스페이스 내에서 새로운 패키지를 생성합니다:

```
cd ~/catkin_ws/src
catkin_create_pkg my_package std_msgs rospy roscpp
```

- **my_package**: 생성할 패키지의 이름.
- **std_msgs**: 메시지 전송에 사용되는 표준 메시지 라이브러리 의존성.
- **rospy**: Python용 ROS 라이브러리 의존성.
- **roscpp**: C++용 ROS 라이브러리 의존성.



#### 2. 워크스페이스 빌드

패키지를 생성한 후, 워크스페이스를 다시 빌드합니다:

```
cd ~/catkin_ws
catkin_make
```



#### 3. 패키지 구조 확인

생성된 패키지의 디렉토리 구조를 확인합니다:

```
ls ~/catkin_ws/src/my_package
```

출력 예시:

```
CMakeLists.txt  include  package.xml  src
```



#### 일반화된 패키지 구조

패키지는 일반적으로 다음과 같은 구조를 가집니다:

```
my_package/
├── CMakeLists.txt          # 빌드 시스템 설정 파일
├── package.xml             # 패키지 메타데이터 파일
├── src/                    # 노드의 소스 코드 디렉토리
├── include/                # 헤더 파일 디렉토리 (C++ 프로젝트)
├── scripts/                # 실행 가능한 스크립트 디렉토리
├── msg/                    # 메시지 정의 파일 디렉토리
├── srv/                    # 서비스 정의 파일 디렉토리
└── action/                 # 액션 정의 파일 디렉토리 (선택적)
```





### 패키지의 역할

패키지는 ROS에서 특정 작업을 수행하는 소프트웨어의 모듈화된 단위로, 다음과 같은 특징을 가집니다:

- **확장 가능성**: 새 기능 추가가 용이.
- **독립성**: 각 패키지는 독립적으로 관리 가능.
- **재사용 가능성**: 기존 패키지를 재활용하여 개발 시간 단축.





### 외부 패키지 설치

ROS에서 공식적으로 제공하는 pacakge는 보통 다음과 같은 명령어를 통해 설치됩니다.

```bash
sudo apt-get install ros-noetic-[pacakge name]
```

위 방식으로 설치된 package는 `roscd` 명령어를 통해 폴더에 접근할 수 있습니다.

```bash
roscd [package name]
```

경로에서 확인할 수 있듯이 package를 설치하면 `/opt/ros/noetic/share` 내부에는 알게 모르게 설치된 수많은 package들이 존재합니다.
`vs code`를 통해 프로그램 코드를 모두 확인할 수는 있지만, 수정할 수 있는 권한이 없습니다.

![image](https://user-images.githubusercontent.com/91526930/235362934-a74b67f4-0026-4bf7-96af-aaeec117a5f3.png)



### 외부 패키지 복사

ROS에서 공식적으로 만들어서 제공하는 package는 아니지만, 특정 회사나 개인이 만들어 package를 github를 통해 배포하는 경우가 있습니다.

- ROS 인증(?) 받은 package의 경우, 아래와 같이 설치 가능

  ```bash
  sudo apt-get install ros-noetic-[pacakge name]
  ```

- 그러나, 해당 package를 직접 수정하여 build하고자 한다면 다음과 같이 `~/catkin_ws/src` 내부에 패키지를 복사 가능.
  (catkin_ws 내부에 복사된 패키지는 편집이 가능함)

  ```bash
  Location: ~/catkin/src
  git clone https://github.com/[USERNAME]/[REPOSITORY_NAME].git
  git clone -b [branch name] https://github.com/[USERNAME]/[REPOSITORY_NAME].git # 특정 branch를 복사해야 하는 경우
  ```

- 패키지 빌드

  ```bash
  Location: ~/catkin
  catkin_make
  ```



# Concept and Creation of a Package

## What is a Package?

In ROS, a **package** is the basic unit of software modules that performs functional roles within the ROS system:

- **Data Integration**: Efficiently manages various data flows through nodes, message files, and service files.  
- **System Modularization**: Breaks down complex robotic systems into independent software modules for easier design and maintenance.  
- **Support for Scalability**: Extend robot functionality by combining existing and new packages.  

A package typically contains the following elements:

- **Node**: Executable programs in ROS.  
- **Message Definition Files (msg)**: Define the data structures exchanged between nodes.  
- **Service Definition Files (srv)**: Define request-response data structures.  
- **Executables**: Compiled binaries or script files.  

Packages can be managed independently within ROS and are designed for reusability.

---

## Package Structure

A package usually consists of the following directories and files:

- **CMakeLists.txt**: Build system configuration file defining the source code and dependencies.  
- **package.xml**: XML file containing package metadata and dependencies.  
- **src/**: Directory containing source code for nodes (Python or C++).  
- **include/**: Directory for header files (used in C++ projects).  
- **scripts/**: Directory containing executable Python scripts.  
- **msg/**: Directory containing user-defined message files.  
- **srv/**: Directory containing user-defined service files.  
- **action/**: Directory containing action files (optional).  

---

## Practice: Creating a Package

### 1. Create a Package
Within the workspace, create a new package:

```bash
cd ~/catkin_ws/src
catkin_create_pkg my_package std_msgs rospy roscpp
```

- **my_package**: The name of the package to create.
- **std_msgs**: Dependency on the standard message library.
- **rospy**: Dependency for Python ROS library.
- **roscpp**: Dependency for C++ ROS library.



#### 2. Build the Workspace

After creating the package, rebuild the workspace:

```
cd ~/catkin_ws
catkin_make
```



#### 3. Check Package Structure

Verify the directory structure of the created package:

```
ls ~/catkin_ws/src/my_package
```

Example output:

```
CMakeLists.txt  include  package.xml  src
```



#### Generalized Package Structure

```
my_package/
├── CMakeLists.txt          # Build system configuration file
├── package.xml             # Package metadata file
├── src/                    # Source code for nodes
├── include/                # Header files (C++ projects)
├── scripts/                # Executable scripts
├── msg/                    # Message definition files
├── srv/                    # Service definition files
└── action/                 # Action definition files (optional)
```





### Role of a Package

A package in ROS is a modular software unit that performs specific tasks with the following characteristics:

- **Scalability**: Easy to add new functionality.
- **Independence**: Each package can be managed independently.
- **Reusability**: Existing packages can be reused to reduce development time.





### Installing External Packages

Official ROS packages can typically be installed using:

```bash
sudo apt-get install ros-noetic-[pacakge name]
```

Installed packages can be accessed with the `roscd` command:

```bash
roscd [package name]
```

As seen in the path, packages installed via apt are located under:
`/opt/ros/noetic/share`

These packages can be inspected with editors like `VS Code`, but modification permissions are not granted.

![image](https://user-images.githubusercontent.com/91526930/235362934-a74b67f4-0026-4bf7-96af-aaeec117a5f3.png)



### Copying External Packages

Sometimes, packages are developed and distributed by companies or individuals via GitHub rather than being officially provided by ROS.

- If the package is ROS-certified, it can be installed as:

  ```bash
  sudo apt-get install ros-noetic-[pacakge name]
  ```

- However, if you need to modify and build the package, you can copy it into `~/catkin_ws/src`.
(Packages inside `catkin_ws` are editable.)

  ```bash
  Location: ~/catkin/src
  git clone https://github.com/[USERNAME]/[REPOSITORY_NAME].git
  git clone -b [branch name] https://github.com/[USERNAME]/[REPOSITORY_NAME].git # 특정 branch를 복사해야 하는 경우
  ```

- Build the Package

  ```bash
  Location: ~/catkin
  catkin_make
  ```

  
