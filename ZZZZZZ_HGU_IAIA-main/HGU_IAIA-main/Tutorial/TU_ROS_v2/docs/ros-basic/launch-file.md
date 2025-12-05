# Launch 파일(.launch)

### 개념 설명

ROS의 **Launch 파일**은 여러 노드를 동시에 실행하고 설정을 관리할 수 있도록 도와주는 XML 기반의 스크립트 파일입니다. 복잡한 ROS 시스템에서 필수적인 구성 요소로, 실행 및 관리의 편의성을 제공합니다.

#### **Launch 파일의 주요 특징**

- **노드 실행**: 여러 노드를 한 번에 실행할 수 있습니다.
- **파라미터 설정**: 노드에 필요한 파라미터를 설정하거나 값을 전달할 수 있습니다.
- **의존성 관리**: 노드 간의 실행 순서를 조정하고 필요한 설정을 적용할 수 있습니다.
- **재사용성**: 동일한 작업을 반복적으로 실행할 때, 동일한 파일을 사용할 수 있습니다.



#### **Launch 파일의 주요 목적**

1. 복잡한 ROS 시스템에서 다수의 노드 실행을 자동화.
2. 파라미터와 설정 값을 노드에 전달하여 유연한 실행 환경 제공.
3. 실수로 인한 실행 오류를 줄이고 작업 속도 향상.



### Launch 파일 기본 구조

Launch 파일은 XML 형식으로 작성되며, `<launch>` 태그 안에 실행할 노드 및 설정을 정의합니다. 기본 구조는 아래와 같습니다:

```
<launch>
    <!-- 공용 파라미터 설정 -->
    <param name="global_param_name" value="global_value" />

    <!-- 노드 실행 -->
    <node pkg="패키지명" type="노드 실행 파일명" name="노드 이름" output="출력 방식">
        <!-- 노드별 파라미터 설정 -->
        <param name="node_param_name" value="node_value" />
    </node>
</launch>
```

- `<launch>`: Launch 파일의 최상위 태그로, 모든 설정을 포함합니다.
- `<param>`: 공용 또는 특정 노드에 전달되는 파라미터를 설정합니다.
  - **공용 파라미터**: Launch 파일 전체에서 사용되는 파라미터.
  - **노드별 파라미터**: 특정 노드에만 전달되는 파라미터.
- `<node>`: 실행할 노드를 정의합니다.
  - `pkg`: 노드가 속한 패키지 이름.
  - `type`: 실행 파일 이름.
  - `name`: 실행되는 노드의 이름.
  - `output`: 출력 방식 (`screen` 또는 `log`).



### 실습: Launch 파일 작성

#### 1. Launch 파일 생성

ROS 워크스페이스의 `launch` 디렉토리를 생성하고 `custom_nodes.launch` 파일을 작성합니다:

```
mkdir -p ~/catkin_ws/src/my_package/launch
cd ~/catkin_ws/src/my_package/launch
touch custom_nodes.launch
```

#### 2. Launch 파일 작성

`custom_nodes.launch` 파일을 열고 아래 내용을 작성합니다:

```
<launch>
    <!-- 공용 파라미터 설정 -->
    <param name="global_param" value="common_value" />

    <!-- 퍼블리셔 노드 실행 -->
    <node pkg="my_package" type="custom_publisher.py" name="publisher_node" output="screen">
        <param name="publish_rate" value="2" />
    </node>

    <!-- 서브스크라이버 노드 실행 -->
    <node pkg="my_package" type="custom_subscriber.py" name="subscriber_node" output="screen" />
</launch>
```

#### 3.  퍼블리셔 수정

```
rate_value = rospy.get_param('~publish_rate', 1)  # 기본값 1 Hz
rate = rospy.Rate(1.0/rate_value)   # Hz
```

#### 4. Launch 파일 실행

Launch 파일을 실행하여 퍼블리셔와 서브스크라이버를 동시에 실행합니다:

```
roslaunch my_package custom_nodes.launch
```

#### 5. 실행 결과 확인

- Launch 파일 실행 후 두 노드가 동시에 실행됩니다.
- 퍼블리셔에서 데이터가 송신되고 서브스크라이버에서 수신되는 로그가 터미널에 출력됩니다.



### 요약

Launch 파일은 여러 노드를 효율적으로 실행하고 관리할 수 있는 도구입니다. 기본 구조와 실습을 통해 공용 파라미터와 노드별 파라미터를 설정하고 활용하는 방법을 익혔으며, 이를 바탕으로 복잡한 ROS 시스템에서도 효율적으로 작업을 수행할 수 있습니다.





# Launch Files (.launch)

## Concept

A **ROS Launch file** is an XML-based script file that allows you to run multiple nodes simultaneously and manage their configurations.  
It is an essential component in complex ROS systems, providing convenience for execution and management.

---

## Key Features of Launch Files
- **Node Execution**: Run multiple nodes at once.  
- **Parameter Settings**: Define or pass parameters needed by nodes.  
- **Dependency Management**: Control the order of execution and apply necessary configurations.  
- **Reusability**: Use the same file to repeatedly execute identical tasks.  

---

## Main Purposes of Launch Files
- Automate the execution of multiple nodes in complex ROS systems.  
- Provide a flexible runtime environment by passing parameters and configuration values to nodes.  
- Reduce execution errors caused by human mistakes and improve workflow speed.  

---

## Basic Structure of a Launch File

A launch file is written in XML format. Nodes and settings are defined inside the `<launch>` tag.  
The basic structure is as follows:

```xml
<launch>
    <!-- Global parameter setting -->
    <param name="global_param_name" value="global_value" />

    <!-- Node execution -->
    <node pkg="package_name" type="executable_name" name="node_name" output="screen">
        <!-- Node-specific parameters -->
        <param name="node_param_name" value="node_value" />
    </node>
</launch>
```

`<launch>`: The top-level tag that includes all configurations.
`<param>`: Defines parameters that can be used globally or for specific nodes.
    - Global Parameters: Used across the entire launch file.
    - Node Parameters: Passed only to a specific node.
`<node>`: Defines a node to be executed.
    - `pkg`: The name of the package where the node belongs.
    - `type`: The name of the executable file.
    - `name`: The name of the node instance being executed.
    - `output`: Defines output mode (`screen` or `log`).



### Practice: Writing a Launch File

#### 1. Create a Launch File

Create a `launch` directory in your ROS workspace and create `custom_nodes.launch`:

```
mkdir -p ~/catkin_ws/src/my_package/launch
cd ~/catkin_ws/src/my_package/launch
touch custom_nodes.launch
```

#### 2. Write the Launch File

Open `custom_nodes.launch` and add the following content:

```
<launch>
    <!-- Global parameter setting -->
    <param name="global_param" value="common_value" />

    <!-- Publisher node -->
    <node pkg="my_package" type="custom_publisher.py" name="publisher_node" output="screen">
        <param name="publish_rate" value="2" />
    </node>

    <!-- Subscriber node -->
    <node pkg="my_package" type="custom_subscriber.py" name="subscriber_node" output="screen" />
</launch>
```

#### 3.  Modify Publisher
 - In the publisher script, modify the rate parameter to use the launch file parameter:

```
rate_value = rospy.get_param('~publish_rate', 1)  # Default: 1 Hz
rate = rospy.Rate(1.0/rate_value)   # Hz
```

#### 4. Run the Launch File

Execute the launch file to run both publisher and subscriber nodes simultaneously

```
roslaunch my_package custom_nodes.launch
```

#### 5. Verify Execution

- After running the launch file, both nodes will be executed at the same time.
- The publisher sends data, and the subscriber receives it.
- Logs will be printed on the terminal to confirm communication



### Summary

A ROS Launch file is a powerful tool to efficiently run and manage multiple nodes.
Through the basic structure and practice, we learned how to set global and node-specific parameters, and how to leverage launch files for streamlined workflows.
This makes working with complex ROS systems much more efficient.
