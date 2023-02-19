# coding: utf-8
import sys
sys.path.append('.')
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
# from common.np import *  # import numpy as np
# from common.util import clip_grads


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model = self.model
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 打乱
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # 计算梯度，更新参数
                outputs = model.forward(batch_x)
                
                # 计算损失值
                loss = model.loss_function(outputs, batch_t)

                model.optimiser.zero_grad()
                loss.backward()
                model.optimiser.step()

                # if max_grad is not None:
                #     clip_grads(grads, max_grad)

                total_loss += loss
                loss_count += 1

                # 评价
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, is_cuda=False):
        self.model = model.cuda() if is_cuda else model
        self.is_cuda = is_cuda
        self.current_epoch = 0
        self.ppl_list = []

    def get_batch(self, x, t, batch_size, time_size):
        if self.is_cuda:
            batch_x = torch.zeros((batch_size, time_size), dtype=torch.long).cuda()
            batch_t = torch.zeros((batch_size, time_size), dtype=torch.long).cuda()
        else:
            batch_x = torch.zeros((batch_size, time_size), dtype=torch.long)
            batch_t = torch.zeros((batch_size, time_size), dtype=torch.long)

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # mini-batch的各笔样本数据的开始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model = self.model
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 获取损失，清空梯度，计算梯度，更新参数
                loss = model(batch_x, batch_t)
                model.optimizer.zero_grad()
                loss.backward()
                # if max_grad is not None:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                model.optimizer.step()

                # 重置RNN的网络状态
                model.reset_state()

                total_loss += loss.item()
                loss_count += 1

                # 评价困惑度
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()