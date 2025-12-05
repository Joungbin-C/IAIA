



# UR5e Hardware Setting

- Control box 전원 켜기
  - 초록색 전원버튼을 누르면, 약 3분 동안 부팅과정을 거친다. 
  - control box의 전원이 완전히 켜지면, 다음의 사진과 같다.

<img src="https://user-images.githubusercontent.com/91526930/234139952-68a2a54d-3a10-4dde-b968-36ebcb89adb1.png" alt="image" style="zoom:50%;" />



- 로봇 프로그램 불러오기

  - 미리 저장되어 있는 로봇 프로그램을 불러온다.
- 로봇 프로그램은 로봇의 동작을 정의하는 것을 말하며, 티칭 펜던트(조작하는 태블릿)에서 프로그램을 작성할 수 있다.
  - 화면 상단에 [열기]를 클릭한다.

  <img src="https://user-images.githubusercontent.com/91526930/234139914-7596143b-bc0c-4dbe-ad50-744752f62f3c.png" alt="image" style="zoom:50%;" />

  - 저장되어 있는 프로그램 중 `ros.urp` 를 선택.
  
  <img src="https://user-images.githubusercontent.com/91526930/234140049-8c69bc7f-fc68-4255-802e-0981ada74a3a.png" alt="image" style="zoom:50%;" />



- 로봇 전원 On
  - 화면의 좌측 하단에 [전원꺼짐] 빨간 버튼을 누르면, 로봇의 상태를 보여준다.
  - 화면 중간에, [켜짐] 초록 버튼을 누르면, 로봇에 전원을 인가한다.
  - [시작] 초록 버튼을 눌러, 로봇을 활성화한다.

<img src="https://user-images.githubusercontent.com/91526930/234140283-fdfdcb14-15f2-44e7-9812-b3434a63759c.png" alt="image" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/91526930/234140311-425d50be-bbef-44c7-9e18-c4fa7a1d82f3.png" alt="image" style="zoom:50%;" />

<img src="https://user-images.githubusercontent.com/91526930/234140351-9ba12fbc-da0b-46c1-85d0-9b40b49614cd.png" alt="image" style="zoom:50%;" />



- 로봇 프로그램
  - 로봇의 동작 시퀀스를 프로그램으로 구성하였으며, 선택한 `ros.urp`프로그램은 다음과 같다.
  - `192.168.0.10` IP 주소를 갖는 외부의 PC로부터 명령을 받아 로봇이 동작을 수행하도록 한다.
  - 스레드(Thread)는 외부 PC의 명령과는 별개로 백그라운드에서 동작하는 것을 의미하며, 
    아래의 예시는, 디지털 출력핀[1]이 True/False 여부에 따라 그리퍼의 동작을 정의한 것이다. 
    ros를 통해 rg2 그리퍼를 직접 제어할 수 없는 상황이기에, 하드웨어적으로 디지털 출력핀의 설정을 통해 그리퍼의 동작을 정의하였다. 

<img src="https://user-images.githubusercontent.com/91526930/234140416-2553909d-5412-4f10-abd1-eb46c30aa5c0.png" alt="image" style="zoom:50%;" />



- PC 연결
  - 랜선으로 PC와 연결한 후, 유선 네트워크를 다음과 같이 설정한다.

<img src="https://user-images.githubusercontent.com/91526930/234139649-6139dcf5-b84a-41f2-9ace-b71725601155.png" alt="Screenshot from 2023-04-24 13-59-34" style="zoom:50%;" />





