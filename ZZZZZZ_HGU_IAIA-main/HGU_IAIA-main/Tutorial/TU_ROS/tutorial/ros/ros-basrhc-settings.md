# basrhc settings



## bashrc 파일

-  terminal 실행마다, 자동으로 초기화하기 위한 파일 script임.

- 초기화 옵션, 환경변수, 함수 등이 기록됨.

- `home` 디렉토리에 존재

- 편집하기
  ```bash
  gedit ~/.bashrc
  ```


## ROS 설치 후,

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5    # libtiff 버전 충돌 방지(ROS <-> Conda)
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

export PYTHONPATH=~/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages
```



## ROS 환경 활성화 함수

```bash
# 자동으로 ROS 환경 활성화
function act_ros {
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:~/catkin_ws/devel/lib/python3/dist-packages
    echo "ROS activated"
}
```





## Anaconda 환경 활성화 함수

```bash
# 자동으로 Anaconda 환경 활성화
function act_conda {
    conda activate base
    export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:~/catkin_ws/devel/lib/python3/dist-packages:~/anaconda3/envs/py38/lib/python3.8/site-packages
    echo "conda activated"
}
```





## 별명 선언 예시

별명 선언을 통해 terminal 내에서 빠른 명령 가능

```bash
alias re='source ~/.bashrc'
alias py38='act_conda && conda activate py38'
```

