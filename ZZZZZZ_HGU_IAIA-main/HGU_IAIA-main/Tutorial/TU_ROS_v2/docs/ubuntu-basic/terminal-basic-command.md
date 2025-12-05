# Ubuntu 터미널에서 자주 사용하는 명령어

Ubuntu의 터미널은 강력한 도구로, 명령어를 통해 시스템을 효율적으로 관리하고 제어할 수 있습니다. 아래는 Ubuntu 터미널에서 주로 사용하는 기본 명령어와 그 설명입니다.

### 1. 디렉토리 탐색 명령어

#### **cd (Change Directory)**

디렉토리를 이동하는 명령어로, 특정 폴더로 위치를 변경할 때 사용합니다.

```
cd /home/user/Documents  # Documents 디렉토리로 이동
cd ..                   # 상위 디렉토리로 이동
cd ~                    # 홈 디렉토리로 이동
cd /                    # 루트 디렉토리로 이동
```



#### **ls (List)**

현재 디렉토리의 파일 및 폴더를 나열하는 명령어입니다.

```
ls                      # 기본 목록 표시
ls -l                   # 자세한 정보 포함
ls -a                   # 숨김 파일 포함
ls -lh                  # 파일 크기를 사람이 읽기 쉬운 형식으로 표시
```




&nbsp;&nbsp;
### 2. 파일 및 디렉토리 관리

#### **mkdir (Make Directory)**

새 디렉토리를 생성합니다.

```
mkdir new_folder        # new_folder라는 이름의 디렉토리 생성
```



#### **rm (Remove)**

파일 또는 디렉토리를 삭제합니다.

```
rm file.txt             # 파일 삭제
rm -r folder            # 폴더와 그 안의 내용 삭제
rm -rf folder           # 강제로 삭제 (주의 필요)
```



#### **cp (Copy)**

파일 또는 디렉토리를 복사합니다.

```
cp file1.txt file2.txt  # file1.txt를 file2.txt로 복사
cp -r folder1 folder2   # folder1의 내용을 folder2로 복사
```



#### **mv (Move)**

파일 또는 디렉토리를 이동하거나 이름을 변경합니다.

```
mv oldname.txt newname.txt  # 파일 이름 변경
mv file.txt /new/path       # 파일을 다른 경로로 이동
```




&nbsp;&nbsp;
### 3. 시스템 관리 명령어

#### **sudo (Super User Do)**

관리자 권한으로 명령어를 실행합니다. 주로 시스템 변경이나 설치 작업에 사용됩니다.

```
sudo apt update         # 패키지 목록 업데이트
sudo rm -rf /folder     # 관리자 권한으로 폴더 삭제
```



#### **apt (Advanced Package Tool)**

패키지 관리 도구로 소프트웨어 설치 및 업데이트에 사용됩니다.

```
sudo apt update         # 패키지 목록 업데이트
sudo apt upgrade        # 시스템 업그레이드
sudo apt install vim    # vim 편집기 설치
sudo apt remove vim     # vim 편집기 제거
```



#### **chmod (Change Mode)**

파일 또는 디렉토리의 권한을 변경합니다.

```
chmod 755 script.sh     # 읽기, 쓰기, 실행 권한 부여
chmod +x script.sh      # 실행 권한 추가
```




&nbsp;&nbsp;
### 4. 파일 보기 및 편집

#### **cat (Concatenate)**

파일 내용을 출력합니다.

```
cat file.txt            # 파일 내용 보기
```



#### **nano**

터미널 기반 텍스트 편집기입니다.

```
nano file.txt           # 파일 열기 및 편집
```


&nbsp;&nbsp;
### 5. 시스템 정보 확인

#### **pwd (Print Working Directory)**

현재 작업 디렉토리의 경로를 출력합니다.

```
pwd                     # 현재 디렉토리 경로 표시
```



#### **df (Disk Free)**

디스크 공간 사용량을 확인합니다.

```
df -h                   # 사람이 읽기 쉬운 형식으로 표시
```



#### **top**

현재 실행 중인 프로세스와 시스템 리소스 사용량을 실시간으로 확인합니다.

```
top                     # 프로세스 모니터링
```




&nbsp;&nbsp;
### 6. 기타 유용한 명령어

#### **clear**

터미널 화면을 지웁니다.

```
clear                   # 화면 정리
```



#### **history**

이전에 실행한 명령어 기록을 확인합니다.

```
history                 # 명령어 기록 보기
```



#### **man (Manual)**

명령어의 사용법을 확인할 수 있는 도움말 페이지를 표시합니다.

```
man ls                  # ls 명령어의 매뉴얼 보기
```





# Frequently Used Ubuntu Terminal Commands

The Ubuntu terminal is a powerful tool that allows efficient system management and control through commands.  
Below are the most commonly used basic commands in the Ubuntu terminal, along with their explanations.

---

## 1. Directory Navigation Commands

### cd (Change Directory)
Moves between directories.  

```bash
cd /home/user/Documents   # Move to the Documents directory
cd ..                     # Move to the parent directory
cd ~                      # Move to the home directory
cd /                      # Move to the root directory
```



#### **ls (List)**

Lists files and folders in the current directory.

```
ls                        # Display basic list
ls -l                     # Display detailed information
ls -a                     # Include hidden files
ls -lh                    # Human-readable file sizes
```




&nbsp;&nbsp;
### 2. File and Directory Management

#### **mkdir (Make Directory)**

Creates a new directory.

```
mkdir new_folder         # Create a directory named new_folder
```



#### **rm (Remove)**

Deletes files or directories.

```
rm file.txt               # Delete a file
rm -r folder              # Delete a folder and its contents
rm -rf folder             # Force delete (use with caution)
```



#### **cp (Copy)**

Copies files or directories.

```
cp file1.txt file2.txt    # Copy file1.txt to file2.txt
cp -r folder1 folder2     # Copy contents of folder1 into folder2
```



#### **mv (Move)**

Moves or renames files/directories.

```
mv oldname.txt newname.txt  # Rename a file
mv file.txt /new/path       # Move a file to a new path
```




&nbsp;&nbsp;
### 3. System Management Commands

#### **sudo (Super User Do)**

Executes commands with administrator privileges, typically for system modifications or installations.

```
sudo apt update           # Update package lists
sudo rm -rf /folder       # Delete folder with admin rights
```



#### **apt (Advanced Package Tool)**

Used for software installation and updates.

```
sudo apt update           # Update package lists
sudo apt upgrade          # Upgrade the system
sudo apt install vim      # Install the vim editor
sudo apt remove vim       # Remove the vim editor
```



#### **chmod (Change Mode)**

Changes file or directory permissions.

```
chmod 755 script.sh       # Grant read, write, execute permissions
chmod +x script.sh        # Add execute permission
```




&nbsp;&nbsp;
### 4. Viewing and Editing Files

#### **cat (Concatenate)**

Displays the contents of a file.

```
cat file.txt              # View file contents
```



#### **nano**

A terminal-based text editor.

```
nano file.txt           # Open and edit file
```


&nbsp;&nbsp;
### 5. System Information

#### **pwd (Print Working Directory)**

Displays the path of the current working directory.

```
pwd                       # Show current directory path
```



#### **df (Disk Free)**

Checks disk space usage.

```
df -h                   # Human-readable format
```



#### **top**

Monitors running processes and system resource usage in real time.

```
top                     # Process monitoring
```




&nbsp;&nbsp;
### 6. Other Useful Commands

#### **clear**

Clears the terminal screen.

```
clear               
```



#### **history**

Shows previously executed commands.

```
history               
```



#### **man (Manual)**

Displays the manual page (help) for a command.

```
man ls                 # Show the manual for the ls command
```

