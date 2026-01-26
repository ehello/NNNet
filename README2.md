# scramble4gpu.py
## Workers
- 针对每个GPU，如果 free memory/total memory > proportion则被选中，占用free memory * 0.9 的显存跑matmul



# gpu_guardian.py 后台程序

## 触发条件
- 检测到最近1小时内多卡平均GPU利用率小于40%

## Workers
- 针对每个GPU，占用 free memory * 0.9 的显存跑 matmul
- worker 如果被 `lsof -t /dev/nvidia* | xargs -r kill -9` 或 `fuser -v /dev/nvidia* | awk '{print $NF}' | xargs -I {} kill -9 {}` 杀死，gpu_guardian.py 后台程序还在继续运行，持续监控，注意只有当 gpu_id 不在 self.workers 字典里时才会启动新 worker。

## 日志输出

### 监控
- 程序启动：`GPU Guardian 启动 - 阈值: 40%, 窗口: 60分钟`
- 异常与退出
    - 查询失败：`查询 GPU 失败: {error}`
    - 手动中断：`收到中断信号，退出...`
    - 其他异常：`错误: {error}`
- 触发占用：`GPU {id} 平均利用率 {x}% < 40%

### Worker启动占用
- 启动成功：`启动 worker 占用 GPU {id} (size={n})`
- 被外部杀死：`GPU {id} worker 已被杀死`



