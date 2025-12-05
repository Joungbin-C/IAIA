# 유틸 프로그램 설치

## 	1. terminator 설치

ubuntu의 terminal을 더 편리하게 사용할 수 있는 프로그램이다. 

- Install

```bash
sudo apt update
sudo apt install terminator
```



- Preferences

  - Terminator 실행 후, 우클릭하여 preferences 진입

  - `Keybindings` 탭에서 단축키 설정을 할 수 있다.

    ![image](https://user-images.githubusercontent.com/91526930/234408076-368a309d-d892-4eaa-8d08-6c46c1c9ec07.png)
    ![image](https://user-images.githubusercontent.com/91526930/234409352-b6e1a5b2-175f-4ed0-b5b9-85615b121c9a.png)
    ![image](https://user-images.githubusercontent.com/91526930/234408233-4b967aae-798f-46f6-94d3-bbb5a2a280f5.png)
    ![image](https://user-images.githubusercontent.com/91526930/234409439-7d409b57-2931-4eb7-98c7-4c080197f583.png)


&nbsp;
## 	2. Visual Studio Code 설치

프로그래밍 개발도구로서, 폴더와 파일 목록을 손쉽게 볼 수 있어, ros 입문자에게 매우 편리하다.

- 설치파일 다운로드 `.deb file` -  [vs code](https://code.visualstudio.com/download)

  ![image](https://github.com/user-attachments/assets/f2718160-304f-44f9-81a7-6c2d5b818829)

- unpack

  다운로드한 파일은 `home/Downloads` 경로에 존재한다.

```bash
cd ~/Downloads
sudo dpkg -i code_*.deb
```


&nbsp;
## 	3. github desktop

- 설치파일 다운로드: [Install GithubDesktop](https://github.com/shiftkey/desktop/releases/)
- 설치파일명: GitHubDesktop-linux-amd64-3.4.8-linux1.deb  (2024.12.26 기준)
- unpack

```bash
cd ~/Downloads
sudo dpkg -i GitHubDesktop_*.deb
```


&nbsp;
## 	4. Typora

```bash
wget -qO - https://typora.io/linux/public-key.asc | sudo tee /etc/apt/trusted.gpg.d/typora.asc

# add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update

# install typora
sudo apt-get install typora
```


&nbsp;
## 	5. 한/영 키 설정

- [settings] - [Region & Language] - [Manage Installed Languages] - [install]

  <img src="https://user-images.githubusercontent.com/91526930/234136304-3fa90717-9034-4cff-8337-733da8ebf548.png" alt="image" style="zoom:67%;" />

  ![image](https://user-images.githubusercontent.com/91526930/234136309-d0f575df-d9b0-4e17-8ed6-a4804dac79a2.png)

- 재부팅

```bash
$ reboot
```



- ibus-setup

```bash
$ ibus-setup
```



- [Input Method] -[Add] - [Korean 검색] - [Hangul] - [Add] - [Close]

![](https://user-images.githubusercontent.com/91526930/234136642-6b78a726-7843-493d-958a-b7caf5b5b151.png)

![](https://user-images.githubusercontent.com/91526930/234136663-7fac9277-4909-414a-8281-4367976b06e5.png)

- [settings] - [Region & Language] - [Add an Input Source] - [Korean(Hangul) 선택]

![](https://user-images.githubusercontent.com/91526930/234136729-9456e9ce-d9e6-47fc-9b97-b9da291d2f43.png)

![](https://user-images.githubusercontent.com/91526930/234136739-a2e620f6-cd35-4baf-b9b2-d534fd30d41a.png)

![](https://github.com/user-attachments/assets/2d99737e-58dc-4cd4-be41-28dc409a7920)


# Utility Program Installation

## 1. Install Terminator

A terminal application that makes Ubuntu’s terminal more convenient to use.  

- **Install**

```bash
sudo apt update
sudo apt install terminator
```

- **Preferences**
  - Launch Terminator, right-click anywhere in the window, and open **Preferences**.  
  - In the **Keybindings** tab, you can customize shortcuts.  

  ![image](https://user-images.githubusercontent.com/91526930/234408076-368a309d-d892-4eaa-8d08-6c46c1c9ec07.png)  
  ![image](https://user-images.githubusercontent.com/91526930/234409352-b6e1a5b2-175f-4ed0-b5b9-85615b121c9a.png)  
  ![image](https://user-images.githubusercontent.com/91526930/234408233-4b967aae-798f-46f6-94d3-bbb5a2a280f5.png)  
  ![image](https://user-images.githubusercontent.com/91526930/234409439-7d409b57-2931-4eb7-98c7-4c080197f583.png)  

---

## 2. Install Visual Studio Code

As a development tool, VS Code makes it easy to browse folders and files—very convenient for ROS beginners.  

- **Download installer (.deb file):** [VS Code](https://code.visualstudio.com/download)  

  ![image](https://github.com/user-attachments/assets/f2718160-304f-44f9-81a7-6c2d5b818829)

- **Install**  
  The downloaded file will be located in the `~/Downloads` directory.  

```bash
cd ~/Downloads
sudo dpkg -i code_*.deb
```

---

## 3. GitHub Desktop

- **Download:** [Install GitHub Desktop](https://github.com/shiftkey/desktop/releases/)  
- Example filename: `GitHubDesktop-linux-amd64-3.4.8-linux1.deb` (as of 2024-12-26)  
- **Install**  

```bash
cd ~/Downloads
sudo dpkg -i GitHubDesktop_*.deb
```

---

## 4. Typora

```bash
wget -qO - https://typora.io/linux/public-key.asc | sudo tee /etc/apt/trusted.gpg.d/typora.asc

# add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update

# install typora
sudo apt-get install typora
```

---

## 5. Korean/English Input Settings (Hangul)

- Go to **Settings → Region & Language → Manage Installed Languages → Install**  

  <img src="https://user-images.githubusercontent.com/91526930/234136304-3fa90717-9034-4cff-8337-733da8ebf548.png" alt="image" style="zoom:67%;" />  

  ![image](https://user-images.githubusercontent.com/91526930/234136309-d0f575df-d9e0-4e17-8ed6-a4804dac79a2.png)  

- **Reboot**

```bash
reboot
```

- **Run ibus setup**

```bash
ibus-setup
```

- **Input Method → Add → search “Korean” → Hangul → Add → Close**

![](https://user-images.githubusercontent.com/91526930/234136642-6b78a726-7843-493d-958a-b7caf5b5b151.png)  
![](https://user-images.githubusercontent.com/91526930/234136663-7fac9277-4909-414a-8281-4367976b06e5.png)  

- **Settings → Region & Language → Add an Input Source → select “Korean (Hangul)”**

![](https://user-images.githubusercontent.com/91526930/234136729-9456e9ce-d9e6-47fc-9b97-b9da291d2f43.png)  
![](https://user-images.githubusercontent.com/91526930/234136739-a2e620f6-cd35-4baf-b9b2-d534fd30d41a.png)  
![](https://github.com/user-attachments/assets/2d99737e-58dc-4cd4-be41-28dc409a7920)  



