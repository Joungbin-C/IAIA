# Utility Programs Installation

- Terminator

- Visual Studio Code
- Typora
- Github Desktop





## Terminator

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
  
  
  
  





## Visual Studio Code

프로그래밍 개발도구로서, 폴더와 파일 목록을 손쉽게 볼 수 있어, ros 입문자에게 매우 편리하다.



- 설치파일 다운로드 `.deb file` -  [vs code](https://code.visualstudio.com/download)

  ![image](https://github.com/user-attachments/assets/f2718160-304f-44f9-81a7-6c2d5b818829)



- unpack

  다운로드한 파일은 `home/Downloads` 경로에 존재한다.

```bash
cd ~/Downloads
sudo dpkg -i code_*.deb
```



## Typora

```bash
wget -qO - https://typora.io/linux/public-key.asc | sudo tee /etc/apt/trusted.gpg.d/typora.asc

# add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update

# install typora
sudo apt-get install typora
```





## Github Desktop


- 설치파일 다운로드: [Install GithubDesktop](https://github.com/shiftkey/desktop/releases/)
- 설치파일명: GitHubDesktop-linux-amd64-3.4.3-linux1.deb  (2024.10.14 기준)
- unpack
  
```bash
cd ~/Downloads
sudo dpkg -i GitHubDesktop-linux-amd64-3.4.3-linux1.deb
```

