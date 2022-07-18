# Time    : 2022.07.18 上午 08:53
# Author  : Vandaci(cnfendaki@qq.com)
# File    : timer.py
# Project : ClassicNetwork
import time
import numpy as np


class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


if __name__ == '__main__':
    timer = Timer()
    for i in range(100000):
        i += i
    print(timer.stop())
    pass
