import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import pandas
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(nn.Module):
    # 判别器

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 learning_rate = 0.01,
                 W1 = None, 
                 W2 = None):
        super().__init__()
        
        linear1 = nn.Linear(input_size, hidden_size)
        linear2 = nn.Linear(hidden_size, output_size)

        if W1 is not None and W2 is not None:
            linear1.weight = Parameter(W1)
            linear2.weight = Parameter(W2)
        linear1.bias = Parameter(torch.zeros(hidden_size))
        linear2.bias = Parameter(torch.zeros(output_size))

        # 定义神经网络层
        self.model = nn.Sequential(
            linear1,
            # nn.Sigmoid(),
            nn.LeakyReLU(0.02),
            linear2,
            # nn.Softmax()
        )

        # 初始损失函数
        self.loss_function = nn.CrossEntropyLoss()
        
        # 创建优化器, 使用随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(),lr=learning_rate)
        # self.optimiser = torch.optim.Adam(self.parameters())

        
        # 计数器和进程记录        
        self.counter = 0        
        self.progress = []


    def forward(self, inputs):
        # 直接运行模型
        return self.model(inputs)

    
    def train(self, inputs, targets, progress):
        # 计算网络的输出
        outputs = self.forward(inputs)
        
        # 计算损失值
        loss = self.loss_function(outputs, targets)
        
        # 每训练10次增加计数器
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
        if (self.counter % progress == 0):
            print("counter = ", self.counter)
            
        # 归零梯度，反向传播，更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.3), figsize=(16,8), alpha=0.3, 
            marker='.', grid=True, yticks=(0, 0.25, 0.5, 0.75, 1, 1.25))
        plt.show()