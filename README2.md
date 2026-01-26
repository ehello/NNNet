# scramble4gpu.py
## Workers
- 针对每个GPU，如果 free memory/total memory > proportion则被选中，占用free memory * 0.9 的显存跑matmul



# gpu_guardian.py 后台程序

## 触发条件
- 检测到最近1小时内多卡的平均GPU利用率小于40%

## Workers
- 针对每个GPU，
    - 先进行僵尸进程清理，僵尸进程 为 单卡GPU 利用率 < 10% 且 单卡显存占用 > 30% 的进程
    - 占用 free memory * 0.9 的显存跑 matmul
- worker 如果被 `lsof -t /dev/nvidia* | xargs -r kill -9` 或 `fuser -v /dev/nvidia* | awk '{print $NF}' | xargs -I {} kill -9 {}` 杀死，gpu_guardian.py 后台程序还在继续运行，持续监控，注意只有当 gpu_id 不在 self.workers 字典里时才会启动新 worker。


## 日志输出

### gpu_guardian.py 后台程序
- 程序启动：`GPU Guardian 启动 - 多卡的平均GPU利用率阈值: 40%, 窗口: 60分钟`
- 异常与退出
    - 查询失败：`查询 GPU 失败: {error}`
    - 手动中断：`收到中断信号，退出...`
    - 其他异常：`错误: {error}`
- 触发占用：`GPU {id} 平均利用率 {x}% < 40%，准备占用`



### Worker 生命周期
- 僵尸进程清理
    - 检测到僵尸进程：`GPU {id} 检测到僵尸进程：利用率 {x}%，显存占用 {y}%`
    - 结果
        - 杀死成功：`GPU {id} 已杀死僵尸进程: [pid1, pid2, ...]`
        - 无权限：`无权限杀死进程 {pid}`
- 启动成功：`启动 worker 占用 GPU {id} (size={n})`
- 被外部杀死：`GPU {id} worker 已被杀死`

## 使用方式

```bash
# 默认后台运行（自动杀僵尸进程）
python gpu_guardian.py -t 40 -w 60


# 前台运行（测试）
python gpu_guardian.py -t 40 -w 60 -f
```

## 参数说明
- `-t`：利用率阈值，默认 40%
- `-w`：监控窗口时长，默认 60 分钟
- `-i`：检查间隔，默认 60 秒
- `-f`：前台模式运行（默认为守护进程模式）
- `-l`：日志文件路径，默认 `./gpu_guardian.log`
- `--no-kill-zombie`：禁用自动杀僵尸进程（默认启用）
- `-m`：僵尸进程判定的显存占用阈值，默认 0.3（30%）


