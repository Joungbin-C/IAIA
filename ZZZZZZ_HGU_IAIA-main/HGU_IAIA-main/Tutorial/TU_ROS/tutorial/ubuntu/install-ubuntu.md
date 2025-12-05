# Ubuntu 설치하기 (듀얼부팅)

본 가이드라인은 **Windows가 설치된 PC에 Ubuntu 20.04 OS를 설치**하는 것을 목적으로 함.

**듀얼 부팅(Dual Booting)**: **하나의 컴퓨터에 두 개 이상의 운영 체제를 설치**하여, **컴퓨터를 부팅할 때마다 원하는 운영 체제를 선택**해서 사용할 수 있는 방식을 말함.



**<span style="background-color:#fff5b1">※ 주의사항</span>** : PC 마다 설치 환경이 다르므로, 본 자료와 상이한 부분에 대해서는 **(1) 의심을 갖고** **(2) 충분히 검색 후,** **(3) 안전하게 설치를 진행**할 것. 

[reference] : [ROS 개발환경 구축 - YouTube](https://youtu.be/x7tpah6Tiqw)



## 목차

**1. [준비 작업](##1.-준비-작업)**

​	1.1. [주요 데이터 백업](###1.1.-주요-데이터-백업-(필수!!))

​	1.2. [USB 부팅 디스크 준비](###1.2.-USB-부팅-디스크-준비)

​	1.3. [디스크 파티션 구성](###1.3.-디스크-파티션-구성)

​	1.4. [BIOS 설정 변경](###1.4.-BIOS-설정-변경)

**2. [Ubuntu 설치](##2.-Ubuntu-설치)**

​	2.1. [USB 부팅 및 Ubuntu 설치 창](###2.1.-USB-부팅-및-Ubuntu-설치-창)

​	2.2. [파티션 및 부트로더 설정하기](###2.2.-파티션-및-부트로더-설정하기)

​	2.3. [설치 완료](###2.3.-설치-완료)

**3. [기타](##3.-기타)**

​	3.1. [GPT/MBR 확인 방법](###3.1.-GPT/MBR-확인-방법)

​	3.2. [Ubuntu 설치 후, 듀얼부팅 안되는 문제](###3.2.-Ubuntu-설치-후,-듀얼부팅-안되는-문제)

​	3.3. [Ubuntu 설치 중, Installation type에 파티션이 안보이는 문제](###3.3.-Ubuntu-설치-중,-Installation-type에-파티션이-안보이는-문제)


## 1. 준비 작업

### 1.1. 주요 데이터 백업 (필수!!)

- OS 설치 과정에서 실수로 데이터를 잃을 수 있기 때문에 중요한 데이터는 필수로 백업할 것.

- 외장하드, 클라우드 스토리지 등을 사용해 중요 문서는 백업

#### BitLocker 키 백업 (필수)

<span style="background-color:#fff5b1">OS 설치 과정 중 Windows OS에 피해가 갈 수 있으므로, BitLocker 키를 백업해야함.</span>

- MS 계정으로 로그인하여 [내 계정]-[디바이스] 탭으로 접속

- 등록되어 있는 디바이스의 [Bitlocker 키 보기]를 통해 드라이브에 해당하는 키 ID를 확인 가능.

![Untitled](https://user-images.githubusercontent.com/91526930/233802004-2fe80c3f-2539-46d0-9705-1bf4201c3427.png)

- [운영 체제 드라이브]의 recovery key를 확인하고, 따로 적어둘 것. 같은 PC에 적어두면, 문제 생겼을 시 확인을 할 수 없으므로, **꼭 다른 곳에 적어두어야!**



- BitLocker 끄기

![Untitled2](https://user-images.githubusercontent.com/91526930/233802011-bb7591b6-ab2d-4557-a45b-48e1d0b4762c.png)



[Reference]

https://sol2gram.tistory.com/68

https://www.samsungsvc.co.kr/solution/25356



### 1.2. USB 부팅 디스크 준비

Ubuntu OS를 설치하기 위한 부팅디스크 USB를 준비하는 과정이며, 8GB or 16GB의 공USB를 사용하면 됨.

#### Ubuntu 디스크 이미지(ISO) 파일 다운로드하기

- [ubuntu 20.04.6 LTS](https://releases.ubuntu.com/focal/)

![ubuntu-20.04.6 disk image](https://user-images.githubusercontent.com/91526930/233799641-5a7e2ec0-93f8-427f-a42c-3a3e7925e8f4.png)



#### Rufus 활용하여 USB에 ISO파일 설치하기

- [Rufus 다운로드](https://rufus.ie/ko/)
  - 포터블 다운로드 시 설치 과정 없이 바로 실행 가능

![image](https://user-images.githubusercontent.com/91526930/233799896-4a051f93-a35a-496c-aa30-3db0d592216a.png)



- USB 연결
  - 디스크 이미지 파일 설치 시 USB 내의 파일들은 모두 사라지므로, 다른 저장소에 옮겨둘 것.



- rufus 실행

  - 업데이트 정책 - [아니오]
  - [장치] - USB 선택
  - [부트 유형 ] - [선택] - `ubuntu-(version)-desktop-amd64.iso` 열기![rufus_실행_디스크선택](https://user-images.githubusercontent.com/91526930/233800162-3994bbf3-f50c-4e09-97ef-5d1331e54ae1.png)

  

  - 파티션 구성 GPT/MBR 구분해서 선택.

    - 본인 PC의 GPT/MBR 확인 방법 (문서 하단 참고)

    ![rufus_파티션구성](https://user-images.githubusercontent.com/91526930/233800311-774ef86e-bcda-44cd-a72a-290af0ed118e.png)

    - 파일 시스템
      - GPT : FAT32(기본) 선택
      - MBR: FAT32 / NTFS 둘다 가능  // 특별한 경우가 아니라면, FAT32 추천

  

  - 디스크 이미지 굽기

    ![rufus_이미지 굽기](https://user-images.githubusercontent.com/91526930/233800405-18681b12-3a1b-44dd-b877-a045d1bf5682.png)

    

    - [ISO 이미지 모드로 쓰기(권장)] - OK

    - [데이터 모두 삭제] - OK

    - 완료되면, USB 내 파일 구성이 다음의 결과로 나타남.

      ![rufus_이미지굽기_완료](https://user-images.githubusercontent.com/91526930/233800461-98f5c213-e9b1-4238-8b03-9ae6da951b34.png)



### 1.3. 디스크 파티션 구성

PC에서 사용중인 디스크 파티션 중 Ubuntu OS를 설치할 디스크 공간을 확보하는 과정임.

#### 디스크 공간 확보

- Windows에 있는 디스크 관리 도구를 사용해 Windows 파티션을 축소하여 Ubuntu를 설치할 공간을 만듦.

- [시작] - [컴퓨터 관리] -  [디스크 관리] - [주사용 디스크] - [우클릭] - [볼륨 축소]

- Ubuntu용으로 최소 20GB 이상의 공간 확보할 것.

- 디스크 공간이 작은 경우 이벤트 뷰어로 파일 삭제 / 가상 메모리 제거 / 시스템 보호 사용안함 / BitLocker 제거 확인

[참고자료](https://m.blog.naver.com/toruin84/223388130169)

  ![디스크_볼륨축소](https://user-images.githubusercontent.com/91526930/233800558-1c9838d6-6577-4872-a0b2-c3425e04ce84.png)

  <img src="https://user-images.githubusercontent.com/91526930/233800566-ac8a69e9-357c-4c01-a646-dd435e072f6c.png" alt="디스크_볼륨축소_할당" style="zoom:67%;" />



- 디스크 공간 확보 결과: 할당되지 않은 디스크 볼륨

![02_디스크_볼륨축소_결과](https://user-images.githubusercontent.com/91526930/233800577-49f4cb20-9fda-4434-ba43-8ef54f9cc441.png)



### 1.4. BIOS 설정 변경

#### BIOS 진입

- [Ubuntu OS 설치 USB 연결] - [재부팅] - [BIOS 진입 key 입력]

  - BIOS 진입 key: `Del`, `F2`, `F10`, `F11` 등 제조사별, 제품별 상이함.

    

#### 부팅 옵션 설정

- **부팅순서 (Booting Order)**
  - Ubuntu 설치 USB가 첫 번째로 부팅 되도록 설정할 것.
  - 기존 순서는 Windows OS를 부팅하는 디스크가 1번이 되어 있을 것임. 
- **기타 옵션 설정**
  - Secure Boot : Disable
  - Fast Boot : Disable
  - UEFI mode 선택 (Legacy mode 선택 x)
  - AHCI mode 선택
  - (위 안내대로 설정하면 되나, 특정 PC에서는 설정값이 보이지 않을 수도 있음)



## 2. Ubuntu 설치

### 2.1. USB 부팅 및 Ubuntu 설치 창

- 부팅 순서의 우선순위를 연결된 USB로 설정함.
- 그러면, Ubuntu OS 설치하는 프로그램이 부팅됨.

![image](https://user-images.githubusercontent.com/91526930/233800893-64b75d34-b087-4763-9bf1-b4b16455b678.png)

- Keyboard Layout -  [Continue]

- Updates and other software

  - Normal과 Minimal (상관없음)

  - Install third-party ~~ (선택)

    ![image](https://user-images.githubusercontent.com/91526930/233801007-d312d181-4284-457a-8b35-d2728353bc1f.png)

    

- Installation type

  - Something else 선택

    ![image](https://user-images.githubusercontent.com/91526930/233801028-3089aac4-310d-4416-ac42-340573d5d267.png)

  

### 2.2. 파티션 및 부트로더 설정하기

Ubuntu OS를 설치할 디스크 파티션을 설정하는 과정이며, Windows가 설치된 디스크 파티션도 함께 출력되므로, 주의가 필요함.



#### 파티션 설정창 확인하기

**※주의사항**: PC 사양에 따라, 출력되는 디스크 파티션의 형태가 다르기에, 아래의 예시 그림과 차이가 있을 수 있음. 또한, 아래 작성된 내용은 TA의 경험에 의해 정리된 내용이나, 각 PC의 모든 상황을 고려하지 못하는 한계가 있음. (아래의 설명 내용이 잘못된 것일 수 있으며, 그렇게 판단되는 경우, TA에게 제보할 것)

- 현재까지의 순서 상으로 문제가 없다면, 다음과 같이 **할당되지 않은 공간** `free space`가 출력되어야 함. 

  ![image](https://user-images.githubusercontent.com/91526930/233801431-6a0b2f75-c197-4006-9e1e-9eb346dbe51a.png)

- 위 그림에 대한 부연 설명

  - `/dev/sda` 내에 sda1, sda2, sda3개의 드라이브가 역할에 따라 구분되어 있음.

  - `/dev/sdb` 내에 sdb1개만 있음. 용량을 비교했을 때, ubuntu 설치용으로 만들어 놓은 드라이브임.

  - 위의 PC는 512GB 용량의 디스크가 2개 장착되어 있음

    - `dev/sda` : windows가 설치되어 있음. (windows 부팅 관련한 것들이 포함됨)
    - `dev/sdb` : 아무것도 없음. (포맷된 상태)

  - PC 내 디스크 1개만 장착된 경우,

    - `dev/sda`와 `dev/sdb` 로 구분되지 않고, 동일한 `dev`의 하위항목에 `free space`가 출력될 것임.

    - (예시)

      ```
      /dev/sda
        /dev/sda1 	ntfs
        /dev/sda2    	ntfs
        /dev/sda3    	ntfs
        free space
      ```

    

- 특이사항 발생한 경우 (아래의 그림)

  - Windows에서 할당되지 않은 공간을 분리시켰기 때문에 `free space`가 존재해야하나, 아래의 상황에서는 매우 작은 공간을 사용한고 있는 것으로 출력함. 이는 PC가 임의로 공간을 할당한 것으로 추정함.
  - 해당 디스크 파티션이 애초 `free space`여야 정상인 것을 확신한다면, 해당 공간을 지워 `free space`로 만들어줌.
  - 아래의 `+`  `-`  `Change...` 3개의 버튼 중 `-`를 클릭하여 `free space`로 만들면, 위의 정상적인 화면과 동일하게 됨.

  ![image](https://user-images.githubusercontent.com/91526930/233801135-b4f1a4de-ea0d-4492-a38c-77195029c728.png)





#### 파티션 생성 및 부트로더 설정하기

파티션 및 부트로더 설정 방법은 아래의 2가지 경우에 따라 구분된다.

**Case (1) Windows 부팅 프로그램이 설치된 드라이브와 동일한 드라이브에 Ubuntu OS를 설치하게 되는 경우**

- 보통 PC내 드라이브가 1개인 경우, 이 절차를 따를 수 밖에 없음

- PC 내 드라이브가 2개이더라도, Windows 부팅 프로그램과 동일 드라이브에 Ubuntu OS를 설치할 수는 있으나, 보통은 지양함.

  

**Case (2) Windows 부팅 프로그램이 설치된 드라이브와 구별된 별개의 드라이브에 Ubuntu OS를 설치하게 되는 경우**

- 보통 PC 내 드라이브가 2개인 경우, Windows 용 드라이브와는 구분하여 다른 드라이브에 Ubuntu OS를 설치함.



##### Case (1): Windows 부팅 드라이브 = Ubuntu 설치 드라이브

- root partition 생성하기

  - [`free space`  선택] - [`+` 클릭]  - [Create partion 창]
  - 아래 그림과 같이 설정한 후, `OK`
    - Size: 할당되지 않은 공간 전체
    - Type: Logical
    - Location: Beginning of this space
    - Use as: Ext4 jounaling file system
    - Mount point:  `/` 

  ![image](https://github.com/user-attachments/assets/efb65e8e-4cb3-4380-a1ce-8550a016826c)

  - 파티션 생성 결과 (예상)

      ```
      device 			Type	Mount Point 	...
      --------------------------------------------
      /dev/sda
        /dev/sda1 	ntfs
        /dev/sda2    	ntfs
        /dev/sda3    	ntfs
        /dev/sda4		ext4		/
      ```



- 부트로더 설정하기
  - 하단의 `Device for boot loader installation:` 에서 선택할 드라이브 장치는 1개밖에 없을 것임. 
  - 예를 들면, `/dev/sda  장치명(500GB)` 와 같이 Windows OS가 설치된 드라이브를 선택할 수 밖에 없음.
  - `Install Now` 버튼을 클릭하면, 선택된 디스크가 포맷된다는 경고 및 안내창이 출력됨. 
    - 특이사항이 없다면, `continue`
    - 특이사항이 있다면, 기존의 세팅에서 문제가 없는지 검토 및 검색을 통해 해결해야함.



##### Case (2): Windows 부팅 드라이브 ≠ Ubuntu 설치 드라이브

- Case(1)과는 다르게, 별도의 드라이브에 Ubuntu를 설치하므로 부팅과 관련한 파티션을 생성해야 함.

  - BIOS 설정시 UEFI모드로 선택했으므로, EFI 파티션을 생성하면 됨.

- EFI 파티션 생성하기

  - [`free space`  선택] - [`+` 클릭]  - [Create partion 창]
  - 아래 그림과 같이 설정한 후, `OK`
    - 용량: 512MB 할당
    - Type: Primary
    - Location: Beginning of this space
    - Use as: efi
    - Mount Point:  `입력 x`

- root partition 생성하기

  - [`free space`  선택] - [`+` 클릭]  - [Create partion 창]

  - 아래 그림과 같이 설정한 후, `OK`

    - Size: 할당되지 않은 공간 전체
    - Type: Logical
    - Location: Beginning of this space
    - Use as: Ext4 jounaling file system
    - Mount point:  `/` 

    ![image](https://github.com/user-attachments/assets/efb65e8e-4cb3-4380-a1ce-8550a016826c)
    
  - 결과
	![image](https://user-images.githubusercontent.com/91526930/233802422-f5c6e0a7-8beb-42ce-b0bd-9ee95a5e3b12.png)



- 부트로더 설정하기

  - 하단의 `Device for boot loader installation:` 에서 efi 파티션을 선택할 것.

  - `Install Now` 버튼을 클릭하면, 선택된 디스크가 포맷된다는 경고 및 안내창이 출력됨. 

    - 특이사항이 없다면, `continue`
    - 특이사항이 있다면, 기존의 세팅에서 문제가 없는지 검토 및 검색을 통해 해결해야함.
    - 아래 그림과 같은 상황에서 진행하게 된다면, efi 파티션 관련한 에러가 출력될 것임.

    ![image](https://user-images.githubusercontent.com/91526930/233801651-2ff7a1cd-072b-4d99-9806-e84906008c5e.png)

  

  - 정상적인 안내창

    ![image](https://user-images.githubusercontent.com/91526930/233801734-1cc16c7a-5714-4a97-9df6-1921901a262a.png)



### 2.3. 설치 완료

- [재부팅] - [USB 제거] - [Set PC name & password]
  - 비밀번호는 짧게 하는게 편함.

![image](https://user-images.githubusercontent.com/91526930/233801833-1f2f3665-e533-4796-91ed-23a307837df5.png)





## 3. 기타

### 3.1. GPT/MBR 확인 방법

- [시작] - [컴퓨터 관리] - [디스크 관리] - [디스크 우클릭] - [속성] - [볼륨 탭]

![컴퓨터관리_디스크속성](https://user-images.githubusercontent.com/91526930/233802295-ff2a8325-bb08-4683-b830-c57ca0245a18.png)

![디스크속성_파티션형식](https://user-images.githubusercontent.com/91526930/233802314-119508c8-ec6a-4d98-91f7-7b5ff453d33e.png)



[Reference]

https://reason1241.tistory.com/15



### 3.2. Ubuntu 설치 후, 듀얼부팅이 안되는 문제

부팅할 때, 윈도우/우분투 선택 창이 출력되지 않고, 바로 윈도우로 진입하는 경우

- 수동으로 부팅 메뉴 진입하여, 부팅 선택하기

  |  제조사  |               BIOS               |            Boot Menu            |
  | :------: | :------------------------------: | :-----------------------------: |
  | 삼성전자 |                F2                |        F10 / 일부는 ESC         |
  |  LG전자  |                F2                |     F10 (노트북) / F12(PC)      |
  |   한성   |                F2                |               F7                |
  |  TG삼보  |            F2 or Del             |               F12               |
  | 주연테크 |            F2 or Del             |               F7                |
  |    HP    | F10 or ESC(startup menu) and F10 | F9 or ESC(startup menu) and F10 |
  |  레노버  |          F2 / 일부는 F1          |               F12               |
  |   DELL   |                F2                |               F12               |
  |   ASUS   |            F2 or Del             |          ESC or F8(PC)          |
  |   MSI    |               Del                |               F11               |
  |   ACER   |                F2                |               F12               |
  | GIGABYTE |            F2 or Del             |               F12               |
  | TOSHIBA  |      F2 / 일부는 ESC and F1      |               F12               |
  |  INTEL   |                F2                |               F10               |


### 3.3. ubuntu 설치 중 Installation type에 파티션이 안보이는 문제
이는 Intel CPU 13세대 이후에 적용되는 **VMD Controller**가 활성화되어 있기 때문

- 바이오스 advantaged mode에서 VMD Controller 비활성화
- Ubuntu 20.04 설치
- 설치 후 Ubuntu가 정상 설치된 것으로 확인되면 VMD Controller 활성화(활성화 안할 시 윈도우OS 블루스크린 화면 뜸)  
