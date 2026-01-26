# -*- coding: utf-8 -*-
"""
GPU Guardian - 后台守护进程
监控 GPU 使用率，当 1 小时内平均利用率 < 40% 时自动占用
worker 被杀死后守护进程继续运行
"""
import os
import sys
import time
import signal
import argparse
import multiprocessing
from collections import deque
from datetime import datetime

import numpy as np

try:
    import torch
except ImportError:
    try:
        import tensorflow as tf
    except ImportError:
        print("需要安装 pytorch 或 tensorflow")
        sys.exit(1)


def query_gpu_utilization():
    """查询所有 GPU 的利用率"""
    cmd = 'nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits'
    try:
        results = os.popen(cmd).readlines()
        gpu_data = []
        for line in results:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 3:
                gpu_data.append({
                    'index': int(parts[0]),
                    'utilization': int(parts[1]),
                    'memory_free': int(parts[2])
                })
        return gpu_data
    except Exception as e:
        print(f"查询 GPU 失败: {e}")
        return []


def compute_storage_size(memory_mb):
    """计算可占用的张量大小 = free memory * 0.9 """
    return int(pow(memory_mb * 1024 * 1024 / 8, 1/3) * 0.9)


def worker(gpu_id, size):
    """GPU 占用 worker，会被外部 kill 命令杀死"""
    try:
        device = f'cuda:{gpu_id}'
        a = torch.zeros([size, size, size], dtype=torch.double, device=device)
        while True:
            torch.mul(a[0], a[0])
            time.sleep(0.1)
    except Exception:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        a = tf.zeros([size, size, size], dtype=tf.dtypes.float64)
        while True:
            tf.matmul(a[0], a[0])
            time.sleep(0.1)


class GPUGuardian:
    def __init__(self, threshold=40, window_minutes=60, check_interval=60):
        self.threshold = threshold  # 利用率阈值
        self.window_minutes = window_minutes  # 监控窗口（分钟）
        self.check_interval = check_interval  # 检查间隔（秒）
        
        # 每个 GPU 的历史利用率记录
        self.history = {}  # {gpu_id: deque of (timestamp, utilization)}
        self.workers = {}  # {gpu_id: Process}
        self.running = True
        
        # 忽略子进程退出信号，防止僵尸进程
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    
    def log(self, msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {msg}", flush=True)
    
    def update_history(self, gpu_data):
        """更新 GPU 利用率历史"""
        now = time.time()
        cutoff = now - self.window_minutes * 60
        
        for gpu in gpu_data:
            gpu_id = gpu['index']
            if gpu_id not in self.history:
                self.history[gpu_id] = deque()
            
            # 添加新记录
            self.history[gpu_id].append((now, gpu['utilization']))
            
            # 清理过期记录
            while self.history[gpu_id] and self.history[gpu_id][0][0] < cutoff:
                self.history[gpu_id].popleft()
    
    def get_avg_utilization(self, gpu_id):
        """获取指定 GPU 在窗口期内的平均利用率"""
        if gpu_id not in self.history or len(self.history[gpu_id]) == 0:
            return None
        return np.mean([u for _, u in self.history[gpu_id]])
    
    def cleanup_dead_workers(self):
        """清理已死亡的 worker 进程"""
        dead = []
        for gpu_id, proc in self.workers.items():
            if not proc.is_alive():
                dead.append(gpu_id)
                self.log(f"GPU {gpu_id} worker 已被杀死")
        
        for gpu_id in dead:
            del self.workers[gpu_id]
    
    def spawn_worker(self, gpu_id, memory_free):
        """启动新的 worker 占用 GPU"""
        size = compute_storage_size(memory_free)
        proc = multiprocessing.Process(target=worker, args=(gpu_id, size), daemon=True)
        proc.start()
        self.workers[gpu_id] = proc
        self.log(f"启动 worker 占用 GPU {gpu_id} (size={size})")
    
    def run(self):
        """主循环"""
        self.log(f"GPU Guardian 启动 - 阈值: {self.threshold}%, 窗口: {self.window_minutes}分钟")
        
        while self.running:
            try:
                # 清理死掉的 worker
                self.cleanup_dead_workers()
                
                # 查询 GPU 状态
                gpu_data = query_gpu_utilization()
                if not gpu_data:
                    time.sleep(self.check_interval)
                    continue
                
                # 更新历史记录
                self.update_history(gpu_data)
                
                # 检查每个 GPU
                for gpu in gpu_data:
                    gpu_id = gpu['index']
                    avg_util = self.get_avg_utilization(gpu_id)
                    
                    if avg_util is None:
                        continue
                    
                    # 如果平均利用率低于阈值且没有 worker 在运行
                    if avg_util < self.threshold and gpu_id not in self.workers:
                        # 确保收集了完整窗口期的数据
                        min_samples = self.window_minutes * 60 // self.check_interval
                        if len(self.history[gpu_id]) >= min_samples:
                            self.log(f"GPU {gpu_id} 平均利用率 {avg_util:.1f}% < {self.threshold}%，启动占用")
                            self.spawn_worker(gpu_id, gpu['memory_free'])
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.log("收到中断信号，退出...")
                self.running = False
            except Exception as e:
                self.log(f"错误: {e}")
                time.sleep(self.check_interval)
        
        # 清理所有 worker
        for proc in self.workers.values():
            if proc.is_alive():
                proc.terminate()


def daemonize():
    """将进程变为守护进程"""
    if os.fork() > 0:
        sys.exit(0)
    
    os.setsid()
    
    if os.fork() > 0:
        sys.exit(0)
    
    # 重定向标准输入输出
    sys.stdout.flush()
    sys.stderr.flush()
    
    with open('/dev/null', 'r') as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())


def main():
    parser = argparse.ArgumentParser(description='GPU Guardian - GPU 使用率监控守护进程')
    parser.add_argument('-t', '--threshold', type=int, default=40,
                        help='GPU 利用率阈值 (默认: 40%)')
    parser.add_argument('-w', '--window', type=int, default=60,
                        help='监控窗口时长，分钟 (默认: 60)')
    parser.add_argument('-i', '--interval', type=int, default=60,
                        help='检查间隔，秒 (默认: 60)')
    parser.add_argument('--foreground', '-f', action='store_true',
                        help='前台模式运行（默认为守护进程模式）')
    parser.add_argument('-l', '--log', type=str, default="./gpu_guardian.log",
                        help='日志文件路径 (守护模式下建议指定)')
    args = parser.parse_args()
    
    if not args.foreground:
        daemonize()
        if args.log:
            log_file = open(args.log, 'a', buffering=1)
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())
    
    guardian = GPUGuardian(
        threshold=args.threshold,
        window_minutes=args.window,
        check_interval=args.interval
    )
    guardian.run()


if __name__ == '__main__':
    main()

"""
后台守护进程运行（默认）
# 脱离终端，关闭终端进程也不死，只能通过 kill 或者 pkill -f gpu_guardian.py 杀死
    python gpu_guardian.py -t 40 -w 60 
前台运行（测试）
    python gpu_guardian.py -t 40 -w 60  -f
"""