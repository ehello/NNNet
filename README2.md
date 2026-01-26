# scramble4gpu.py
- workers:
    - 针对每个GPU，如果 free memory/total memory > proportion则被选中，占用free memory * 0.9 的显存跑matmul



# gpu_guardian.py 后台程序

- 触发信号：
    - 检测到最近1小时内多卡平均GPU小于40%

- workers:
    - 针对每个GPU，如果 free memory/total memory > proportion则被选中，占用free memory * 0.9 的显存跑matmul

    - 这个worker如果被 `lsof -t /dev/nvidia* | xargs -r kill -9` 或者 `fuser -v /dev/nvidia* | awk '{print $NF}' | xargs -I {} kill -9 {}` 杀死，gpu_guardian.py 后台程序还要保持在，持续监控