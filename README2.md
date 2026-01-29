# scramble4gpu.py
## Workers
- 针对每个GPU，如果 free memory/total memory > proportion则被选中，占用 free memory * ratio^3 的显存跑matmul（ratio 默认 0.6，约22%）



# gpu_guardian.py 后台程序

## 触发条件
- 程序首次启动时，立即占用所有 GPU
- 窗口时间内多卡平均利用率 < 30%（计算方式：各GPU窗口内平均利用率 → 再取多卡均值）

## Workers
- 针对每个GPU，
    - 先进行僵尸进程清理(单卡GPU 利用率 < 10% 且 单卡显存占用 > 30% 的进程)。
    - 占用 free memory * ratio^3 的显存跑 matmul（ratio 默认 0.6，约22%）
- worker 如果被 `lsof -t /dev/nvidia* | xargs -r kill -9` 或 `fuser -v /dev/nvidia* | awk '{print $NF}' | xargs -I {} kill -9 {}` 杀死，gpu_guardian.py 后台程序还在继续运行，持续监控，注意只有当 gpu_id 不在 self.workers 字典里时才会启动新 worker。




## 日志输出

### gpu_guardian.py 后台程序
- 程序启动：`GPU Guardian 启动 - 窗口: {w}s, 多卡平均利用率阈值: {t}%, 空闲显存占用比例系数: {ratio}`
- 异常与退出
    - 查询失败：`查询 GPU 失败: {error}`
    - 手动中断：`收到中断信号，退出...`
    - 其他异常：`错误: {error}`
- 触发占用：
    - 首次启动占用：`首次启动，立即占用所有 GPU`
    - `窗口 HH:MM:SS-HH:MM:SS ({w}s) 内各GPU平均利用率: [GPU0: {x}%, GPU1: {y}%, ...], 多卡平均: {z}% < {t}%, 空闲显存占用比例系数: {ratio}，开始占用空闲 GPU`
- 定时日志（每隔窗口时长打印一次）：
    - `定时打印日志：窗口 HH:MM:SS-HH:MM:SS ({w}s) 内各GPU平均利用率: [GPU0: {x}%, GPU1: {y}%, ...], 多卡平均: {z}%`



### Worker 生命周期
- 僵尸进程清理
    - 检测到僵尸进程：`GPU {id} 检测到僵尸进程：利用率 {x}%，显存占用 {y}%`
    - 结果
        - 杀死成功：`GPU {id} 已杀死僵尸进程: [pid1, pid2, ...]`
        - 无权限：`无权限杀死进程 {pid}`
- 启动成功：`启动 worker 占用 GPU {id} (size={n})`
- 被外部杀死：`GPU {id} worker 已被杀死`

## 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t, --threshold` | 多卡平均 GPU 利用率阈值 | 30% |
| `-w, --window` | 监控窗口时长（秒） | 3600 |
| `-i, --interval` | 检查间隔（秒） | 1 |
| `-d, --daemon` | 守护模式: 0=前台, 1=后台 | 1 |
| `-l, --log` | 日志文件路径 | ./gpu_guardian.log |
| `-r, --ratio` | 空闲显存占用比例系数 | 0.6 |
| `-m, --zombie-memory` | 僵尸进程显存占用阈值 | 0.3 |
| `--no-kill-zombie` | 禁用自动杀僵尸进程 | - |