#这里相当于改个文件名，保守点; 用于在平台提交的命令行，注意一定要 -d 0, 否则它不阻塞

set -x ; apt update ; apt install git -y ; [ -d NNNet ] || git clone https://github.com/ehello/NNNet.git ; cd NNNet ; python gpu_guardian.py -d 0
