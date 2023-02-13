import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import pandas
import numpy as np
import matplotlib.pyplot as plt


class SimpleCBOW(nn.Module):

    def __init__(self, 
                 vocab_size, 
                 hidden_size, 
                 learning_rate = 0.01,
                 W1 = None, 
                 W2 = None):
        super().__init__()
        

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # 初始损失函数
        self.loss_function = nn.CrossEntropyLoss()
        
        # 创建优化器, 使用随机梯度下降
        # self.optimiser = torch.optim.SGD(self.parameters(),lr=learning_rate)
        self.optimiser = torch.optim.Adam(self.parameters())

        
        # 计数器和进程记录        
        self.counter = 0
        self.total_loss = 0
        self.loss_count = 0  
        self.loss_list = []



    def forward(self, inputs):
        x = self.embedding(inputs).sum(dim=1)
        x = self.linear(x)
        return x

    
    def train(self, inputs, targets, iters, epoch, max_iters):
        # 计算网络的输出
        outputs = self.forward(inputs)
        
        # 计算损失值
        loss = self.loss_function(outputs, targets)
        
            
        # 归零梯度，反向传播，更新权重
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        
        self.total_loss += loss
        self.loss_count += 1


        # 定期输出学习过程
        if (iters+1) % 10 == 0:
            avg_loss = self.total_loss / self.loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            self.loss_list.append(avg_loss.detach().numpy())
            self.total_loss, self.loss_count = 0, 0


    def plot_progress(self):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label='train')
        plt.xlabel('iterations (x10)')
        plt.ylabel('loss')
        plt.show()
