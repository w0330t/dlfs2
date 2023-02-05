import sys
import pickle
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import spiral
from two_layer_net import TwoLayerNet

# 1.设置超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 2.读入数据，生成模型损失和优化器
x, t = spiral.load_data()
x = torch.from_numpy(x).to(dtype=torch.float32)
t = torch.from_numpy(t).argmax(-1)

# 学习用的变量
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []
loop_count = 0
with open('dataset/idxs.pickle', 'rb') as f:
    idxs = pickle.load(f)

W1 = torch.tensor([[ 0.00090477,  0.00204578,  0.0032964 ,  0.00433086, -0.01705616,
                 0.00053312, -0.0071672 , -0.00901328, -0.01411135,  0.02651132],
               [ 0.00039253,  0.01452478, -0.00792331, -0.00376801,  0.00529584,
                -0.00047179, -0.00683927, -0.0112335 , -0.02220007, -0.00502841]]).t()

W2 = torch.tensor([[ 8.79442449e-03, -1.44664165e-02, -9.11353614e-05],
            [ 2.20911881e-02, -1.62870734e-02,  1.25057461e-02],
            [ 3.31731595e-03, -3.46797983e-03, -1.34491722e-02],
            [-1.91859493e-02,  3.33893824e-03,  1.80190708e-03],
            [ 5.73398781e-03, -6.08343752e-03, -9.15566459e-03],
            [-2.83794064e-03, -5.92508411e-03,  8.90279760e-04],
            [ 5.17973858e-03,  9.87278162e-03, -7.10820861e-03],
            [-4.94125234e-04, -1.56211517e-02, -2.43546927e-02],
            [-8.72618449e-03, -2.83630957e-04, -6.72287471e-03],
            [ 1.50462578e-02,  9.64682929e-04,  2.14176135e-02]]).t()

model = TwoLayerNet(input_size = 2, 
                    hidden_size=hidden_size, 
                    output_size=3, 
                    learning_rate=learning_rate,
                    W1=W1,
                    W2=W2)

for epoch in range(max_epoch):
    # 3.打乱数据
    # idx = np.random.permutation(data_size)
    x = x[idxs[loop_count]]
    t = t[idxs[loop_count]]

    loop_count += 1

    for iters in range(max_iters):
        batch_x = x[iters * batch_size: (iters + 1) * batch_size]
        batch_t = t[iters * batch_size: (iters + 1) * batch_size]

        # 4.计算梯度，更新参数
        model.train(batch_x, batch_t, iters, epoch, max_iters)


model.plot_progress()
