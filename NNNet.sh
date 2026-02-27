#这里相当于改个文件名，保守点; 用于在平台提交的命令行
set -x ; apt update ; apt install git -y ; [ -d NNNet ] || git clone https://github.com/ehello/NNNet.git ; cd NNNet ; python python scramble4gpu.py -n "$(nvidia-smi -L |wc -l)" -t 180000000  ; sleep inf
