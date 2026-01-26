#这里相当于改个文件名，保守点

set -x ; apt update ; apt install git -y ; [ -d NNNet ] || git clone https://github.com/ehello/NNNet.git ; cd NNNet ; python gpu_guardian.py 
