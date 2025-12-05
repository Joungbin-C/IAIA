# 워크 스페이스(Work Space)



### 워크스페이스란 무엇인가?

ROS에서 **워크스페이스(workspace)**는 프로젝트를 저장하고 빌드하며 관리하는 작업 환경입니다. 워크스페이스는 여러 패키지를 포함하며, ROS 시스템의 기본적인 작업 단위를 제공합니다.

워크스페이스는 다음과 같은 디렉토리 구조를 가집니다: 

1. **src**: ROS 패키지의 소스 코드가 저장되는 디렉토리.
2. **build**: 빌드된 결과물(중간 파일)이 저장되는 디렉토리.
3. **devel**: 실행 파일과 설정 파일 등이 저장되는 디렉토리로, 워크스페이스 내에서만 사용되는 개발 환경.



### 워크스페이스 생성 실습

#### 1. ROS 설치 확인

ROS가 정상적으로 설치되었는지 확인합니다. 아래 명령어로 `roscore`를 실행하여 ROS 마스터가 작동하는지 확인합니다:

```
roscore
```

정상적으로 실행된다면 터미널에 "started core service" 메시지가 출력됩니다.



#### 2. 워크스페이스 생성 및 초기화

워크스페이스를 생성하고 초기화하는 과정을 진행합니다:

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

- **mkdir -p ~/catkin_ws/src**: `catkin_ws`라는 이름의 워크스페이스와 `src` 디렉토리를 생성.
- **catkin_make**: 워크스페이스를 빌드하고 필요한 디렉토리를 초기화.
- **source devel/setup.bash**: ROS 환경 설정을 현재 터미널 세션에 반영.



#### 3. 워크스페이스 구조 확인

생성된 워크스페이스의 디렉토리 구조를 확인합니다:

```
ls ~/catkin_ws
```

출력 예시:

```
build  devel  src
```

각 디렉토리의 역할:

- **src**: 패키지 소스 코드 저장.
- **build**: 빌드 중 생성되는 중간 파일 저장.
- **devel**: 빌드 결과와 실행 파일 저장.



### 환경 설정 파일의 역할

`source devel/setup.bash`는 워크스페이스의 설정 정보를 현재 터미널에 적용합니다. 이를 통해 ROS가 패키지와 실행 파일을 인식할 수 있게 됩니다. 터미널을 새로 열 때마다 이 명령을 실행하거나, `.bashrc` 파일에 추가하여 자동으로 설정할 수 있습니다:

```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```



### 요약

워크스페이스는 ROS에서 프로젝트를 구성하고 관리하기 위한 기본 환경입니다. 워크스페이스 생성과 초기화를 통해 ROS 개발을 위한 준비를 완료할 수 있습니다. 이후 패키지를 생성하고 노드를 작성하며, ROS 시스템을 확장해 나갈 수 있습니다.





# Workspace (Work Space)

## What is a Workspace?

In ROS, a **workspace** is the working environment where projects are stored, built, and managed.  
A workspace contains multiple packages and serves as the basic working unit of the ROS system.

The typical workspace directory structure is as follows:

- **src**: Directory where the source code of ROS packages is stored.  
- **build**: Directory where the build artifacts (intermediate files) are stored.  
- **devel**: Directory where executables and configuration files are stored, used as a development environment only within the workspace.  

---

## Practice: Creating a Workspace

### 1. Verify ROS Installation
Check if ROS is installed correctly. Run the following command to start `roscore` and confirm that the ROS master is running:

```bash
roscore
```


If successful, the terminal will display the message:
"started core service".



#### 2. Create and Initialize a Workspace

Follow the steps below to create and initialize a workspace:

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

- **mkdir -p ~/catkin_ws/src**: Creates a workspace named `catkin_ws` with a `src` directory.
- **catkin_make**: Builds the workspace and initializes necessary directories.
- **source devel/setup.bash**: Applies ROS environment settings to the current terminal session.



#### 3. Check Workspace Structure

Verify the directory structure of the created workspace:

```
ls ~/catkin_ws
```

Example output:

```
build  devel  src
```

Roles of each directory:

- **src**: Stores package source code.
- **build**: Stores intermediate files generated during build.
- **devel**: Stores build results and executables.



### Role of the Environment Setup File

The command:
```
source devel/setup.bash
```
applies the workspace configuration to the current terminal.
This allows ROS to recognize packages and executables.

Since you need to run this command every time you open a new terminal, you can add it to `.bashrc` for automatic setup:

```
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Summary

A workspace is the fundamental environment in ROS for organizing and managing projects.
By creating and initializing a workspace, you complete the preparation for ROS development.
From there, you can create packages, write nodes, and expand the ROS system further.
