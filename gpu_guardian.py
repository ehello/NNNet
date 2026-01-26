# -*- coding: utf-8 -*-
"""
GPU Guardian - 后台守护进程
监控多卡平均 GPU 利用率，低于阈值时自动占用
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
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    try:
        import tensorflow as tf
    except ImportError:
        print("需要安装 pytorch 或 tensorflow")
        sys.exit(1)


def query_gpu_info():
    """查询所有 GPU 的状态信息"""
    cmd = 'nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits'
    try:
        lines = os.popen(cmd).readlines()
        gpu_list = []
        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                gpu_list.append({
                    'index': int(parts[0]),
                    'utilization': int(parts[1]),
                    'memory_free': int(parts[2]),
                    'memory_total': int(parts[3])
                })
        return gpu_list
    except Exception as e:
        print(f"查询 GPU 失败: {e}")
        return []


def get_gpu_pids(gpu_id):
    """获取指定 GPU 上的进程 PID 列表"""
    cmd = f'nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i {gpu_id}'
    try:
        lines = os.popen(cmd).readlines()
        return [int(line.strip()) for line in lines if line.strip().isdigit()]
    except Exception:
        return []


def kill_pids(pids, log_func):
    """杀死指定的进程列表"""
    killed = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError:
            log_func(f"无权限杀死进程 {pid}")
    return killed


def compute_tensor_size(memory_mb):
    """根据显存计算张量边长 (占用 free memory * 0.9^3≈73% 显存)"""
    return int(pow(memory_mb * 1024 * 1024 / 8, 1/3) * 0.9)


def worker(gpu_id, size):
    """GPU 占用 worker"""
    if USE_TORCH:
        tensor = torch.zeros([size, size, size], dtype=torch.double, device=f'cuda:{gpu_id}')
        while True:
            torch.mul(tensor[0], tensor[0])
            time.sleep(0.1)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        tensor = tf.zeros([size, size, size], dtype=tf.dtypes.float64)
        while True:
            tf.matmul(tensor[0], tensor[0])
            time.sleep(0.1)


class GPUGuardian:
    def __init__(self, threshold, window_minutes, check_interval, kill_zombie, zombie_memory_threshold):
        self.threshold = threshold
        self.window_minutes = window_minutes
        self.check_interval = check_interval
        self.kill_zombie = kill_zombie
        self.zombie_memory_threshold = zombie_memory_threshold
        
        self.history = {}  # {gpu_id: deque of (timestamp, utilization)}
        self.workers = {}  # {gpu_id: Process}
        self.running = True
        self.min_samples = window_minutes * 60 // check_interval
        self.first_run = True  # 首次启动标记
        
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    
    def log(self, msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)
    
    def update_history(self, gpu_list):
        """更新利用率历史记录"""
        now = time.time()
        cutoff = now - self.window_minutes * 60
        
        for gpu in gpu_list:
            gpu_id = gpu['index']
            if gpu_id not in self.history:
                self.history[gpu_id] = deque()
            
            self.history[gpu_id].append((now, gpu['utilization']))
            
            while self.history[gpu_id] and self.history[gpu_id][0][0] < cutoff:
                self.history[gpu_id].popleft()
    
    def get_all_avg_utilization(self):
        """计算多卡整体平均利用率"""
        all_utils = []
        for history in self.history.values():
            all_utils.extend([util for _, util in history])
        return np.mean(all_utils) if all_utils else None
    
    def has_enough_history(self):
        """检查是否收集了足够的历史数据"""
        if not self.history:
            return False
        return all(len(h) >= self.min_samples for h in self.history.values())
    
    def cleanup_dead_workers(self):
        """清理已死亡的 worker"""
        dead_gpus = [gpu_id for gpu_id, proc in self.workers.items() if not proc.is_alive()]
        for gpu_id in dead_gpus:
            self.log(f"GPU {gpu_id} worker 已被杀死")
            del self.workers[gpu_id]
    
    def try_kill_zombie(self, gpu):
        """尝试杀死 指定GPU 上的僵尸进程"""
        gpu_id = gpu['index']
        memory_used_ratio = 1 - gpu['memory_free'] / gpu['memory_total']
        
        if gpu['utilization'] < 10 and memory_used_ratio > self.zombie_memory_threshold:
            self.log(f"GPU {gpu_id} 检测到僵尸进程：指定GPU上的利用率 {gpu['utilization']}%，显存占用 {memory_used_ratio*100:.1f}%")
            killed = kill_pids(get_gpu_pids(gpu_id), self.log)
            if killed:
                self.log(f"GPU {gpu_id} 已杀死僵尸进程: {killed}")
                return True
        return False
    
    def spawn_worker(self, gpu_id, memory_free):
        """启动 worker 占用 GPU"""
        size = compute_tensor_size(memory_free)
        proc = multiprocessing.Process(target=worker, args=(gpu_id, size), daemon=True)
        proc.start()
        self.workers[gpu_id] = proc
        self.log(f"启动 worker 占用 GPU {gpu_id} (size={size})")
    
    def run(self):
        """主循环"""
        self.log(f"GPU Guardian 启动 - 多卡平均利用率阈值: {self.threshold}%, 窗口: {self.window_minutes}分钟")
        
        while self.running:
            try:
                self.cleanup_dead_workers()
                
                gpu_list = query_gpu_info()
                if not gpu_list:
                    time.sleep(self.check_interval)
                    continue
                
                self.update_history(gpu_list)
                
                avg_util = self.get_all_avg_utilization()
                
                # 首次启动或平均利用率低于阈值时触发占用
                should_occupy = self.first_run
                if self.first_run:
                    self.log("首次启动，立即占用所有 GPU")
                    self.first_run = False
                elif avg_util is not None and self.has_enough_history() and avg_util < self.threshold:
                    self.log(f"多卡平均利用率 {avg_util:.1f}% < {self.threshold}%，开始占用空闲 GPU")
                    should_occupy = True
                
                if should_occupy:
                    for gpu in gpu_list:
                        gpu_id = gpu['index']
                        if gpu_id in self.workers:
                            continue
                        
                        if self.kill_zombie and self.try_kill_zombie(gpu):
                            time.sleep(2)
                            # 重新查询该 GPU 显存
                            for new_gpu in query_gpu_info():
                                if new_gpu['index'] == gpu_id:
                                    gpu = new_gpu
                                    break
                        
                        self.spawn_worker(gpu_id, gpu['memory_free'])
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.log("收到中断信号，退出...")
                self.running = False
            except Exception as e:
                self.log(f"错误: {e}")
                time.sleep(self.check_interval)
        
        for proc in self.workers.values():
            if proc.is_alive():
                proc.terminate()


def daemonize(log_path=None):
    """转为守护进程"""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    
    sys.stdout.flush()
    sys.stderr.flush()
    with open('/dev/null', 'r') as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    
    if log_path:
        log_file = open(log_path, 'a', buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())


def dual_process_guard(target):
    """双进程守护 - 子进程被杀后自动重启"""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    
    while True:
        if os.fork() == 0:
            target()
            sys.exit(0)
        os.wait()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='GPU Guardian - GPU 使用率监控守护进程')
    parser.add_argument('-t', '--threshold', type=int, default=40,
                        help='多卡平均 GPU 利用率阈值 (默认: 40%%)')
    parser.add_argument('-w', '--window', type=int, default=60,
                        help='监控窗口时长，分钟 (默认: 60)')
    parser.add_argument('-i', '--interval', type=int, default=60,
                        help='检查间隔，秒 (默认: 60)')
    parser.add_argument('-d', '--daemon', type=int, default=1, choices=[0, 1, 2],
                        help='守护等级: 0=前台运行, 1=后台守护进程, 2=双进程守护 (默认: 1)')
    parser.add_argument('-l', '--log', type=str, default='./gpu_guardian.log',
                        help='日志文件路径')
    parser.add_argument('--no-kill-zombie', action='store_true',
                        help='禁用自动杀僵尸进程')
    parser.add_argument('-m', '--zombie-memory', type=float, default=0.3,
                        help='僵尸进程判定的显存占用阈值 (默认: 0.3)')
    args = parser.parse_args()
    
    guardian = GPUGuardian(
        threshold=args.threshold,
        window_minutes=args.window,
        check_interval=args.interval,
        kill_zombie=not args.no_kill_zombie,
        zombie_memory_threshold=args.zombie_memory
    )
    
    if args.daemon == 0:
        # Level 0: 前台运行
        guardian.run()
    elif args.daemon == 1:
        # Level 1: 后台守护进程
        daemonize(args.log)
        guardian.run()
    else:
        # Level 2: 双进程守护
        daemonize(args.log)
        dual_process_guard(guardian.run)

if __name__ == '__main__':
    main()

"""
守护等级说明:
  -d 0  前台运行（测试用）
  -d 1  后台守护进程（默认），kill pid 或者 pkill -f gpu_guardian.py 可杀
  -d 2  双进程守护，需要连续 pkill 才能杀干净 for i in {1..5}; do pkill -9 -f gpu_guardian.py; sleep 0.2; done

示例:
    python gpu_guardian.py -d 0 -t 40 -w 60      # 前台运行
    python gpu_guardian.py -d 1 -t 40 -w 60      # 后台守护
    python gpu_guardian.py -d 2 -t 40 -w 60      # 双进程守护（最难杀）
"""