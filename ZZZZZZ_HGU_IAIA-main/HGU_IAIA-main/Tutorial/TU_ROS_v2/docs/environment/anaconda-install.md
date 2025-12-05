# Anaconda 설치

## 1. Anaconda Installer

- Download anaconda Installer: [link](https://www.anaconda.com/download/success)
  ![](https://github.com/user-attachments/assets/1a60754e-35f1-4c9d-b42f-154df6e332ef)


&nbsp;
## 2. Execute Installer File (shell script)

```bash
cd ~/Downloads			  # 다운로드 위치로 이동
sudo bash Anaconda3_*.sh	# shell script 실행
```

- 프로그램 설치 약관에 대한 내용 출력시, Enter를 계속해서 누르고 있으면 됨
- 묻는 질문에는 모두 yes 입력

- When the installation license agreement is displayed, press Enter repeatedly to proceed.
- For all questions during the installation, type 'yes'.

[reference link](https://record-everything.tistory.com/entry/Ubuntu-2004-%EC%9A%B0%EB%B6%84%ED%88%AC%EC%97%90-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-%EC%84%A4%EC%B9%98-%EB%B0%8F-Python-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95)


&nbsp;
## 3. Initialize bashrc 

```bash
source ~/.bashrc
```

- 프롬프트를 실행할 때마다 디폴트로 conda 환경이 자동으로 활성화되면서, 터미널에 항상 `(base)`가 함께 출력됨

- 터미널을 열때마다 conda 환경을 활성화되는 것을 막기 위해 아래 명령어 입력

- By default, the conda base environment is automatically activated each time a terminal is opened, showing (base) in the prompt.

- To prevent conda from activating automatically when opening the terminal, run the following command:

```bash
conda config --set auto_activate_base false
```

