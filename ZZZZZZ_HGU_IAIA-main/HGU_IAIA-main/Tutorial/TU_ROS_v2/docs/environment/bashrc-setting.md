# bashrc Settings

## 1. bashrc File

- terminal 실행마다, 자동으로 초기화하기 위한 파일 script임.

- 초기화 옵션, 환경변수, 함수 등이 기록됨.

- `home` 디렉토리에 존재

- 편집하기

- A script file that runs automatically every time the terminal is launched.  
- Stores initialization options, environment variables, and functions.  
- Located in the `home` directory.
- To edit:
  

  ```bash
  gedit ~/.bashrc
  ```


&nbsp;
## 2. After Installing ROS

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5    # libtiff 버전 충돌 방지(ROS <-> Conda)
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

export PYTHONPATH=~/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
```


&nbsp;
## 3. ROS 환경 활성화 [ROS Environment Activation Function]

```bash
# 자동으로 ROS 환경 활성화
# Automatically activate ROS environment
function act_ros {
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:~/catkin_ws/devel/lib/python3/dist-packages
    echo "ROS activated"
}
```




&nbsp;
## 4. Anaconda 환경 활성화 함수 [Anaconda Environment Activation Function]

```bash
# 자동으로 Anaconda 환경 활성화
# Automatically activate Anaconda environment
function act_conda {
    conda activate base
    export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:~/catkin_ws/devel/lib/python3/dist-packages:~/anaconda3/envs/py38/lib/python3.8/site-packages
    echo "conda activated"
}
```




&nbsp;
## 5. 별명 선언 예시 [Alias Examples]

별명 선언을 통해 terminal 내에서 빠른 명령 가능
Aliases allow quick commands in the terminal

```bash
alias re='source ~/.bashrc'
alias py38='act_conda && conda activate py38'
```

