# Ubuntu - Basic Command

- man

  - manual

  - 특정 명령어에 대해 자세히 알고 싶을 때, 도움말로 이동

    ```bash
    $ man ls
    $ man cd
    ```

  

- ls

  - list

  - 현재 디렉터리의 내용을 보여줌.

    ```bash
    $ ls
    $ ls -l		# 디렉터리 및 파일의 권한 정보를 포함하여 상세 출력
    $ ls -a		# 디렉터리 및 파일의 숨겨진 정보들까지 모두 출력
    $ ls -la	# 권한 정보와 숨겨진 정보들 모두 출력
    ```



- cd

  - change directory

  - 해당 디렉터리로 이동.

    ```bash
    $ cd ~
    $ cd Desktop
    $ cd ..
    ```



- mkdir

  - make directory

  - 새로운 디렉터리를 생성

    ```bash
    $ mkdir hello_world
    ```



- rm

  - remove

  - 파일이나 디렉터리를 삭제

    ```bash
    $ rm hello_world.sh		# 파일 삭제
    $ rm -f hello_world.sh		# 파일을 삭제할 때, 사용자에게 확인을요구하지 않음
    $ rm -r hello_world		# 디렉터리를 삭제할 때는 -r 옵션. 하위 디렉터리까지 모두 삭제
    ```



- mv

  - move

  - 파일 이름을 변경하거나, 다른 디렉터리로 옮길 때 사용

    ```bash
    $ mv hello_world.sh hello_handong.sh	# 파일 이름 변경
    $ mv hello_world.sh ~/Downloads		# 파일 위치 변경
    ```



- cp

  - copy

  - 파일 또는 디렉터리 복사

    ```bash
    $ cp hello_world.sh ~/Desktop			# 다른 위치에 같은 파일을 복사
    $ cp hello_world.sh hello_handong.sh 		# 작업 위치에 같은 내용의 파일을 새로운 이름으로 복사
    $ cp hello_world.sh ~/Desktop/hello_handong.sh 	# 다른 위치에 같은 내용의 파일을 새로운 이름으로 복사
    ```

    

- alias

  - 자주 사용하는 명령어를 별명으로 정의하기

    ```bash
    $ alias lsa = 'ls -a'	# ls -a 명령어를 lsa 별명으로 입력할 수 있도록 정의
    $ unalias lsa		# lsa 별명 해제
    ```

    