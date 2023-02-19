import sys, torch
sys.path.append('.')

from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
from common.pickel_weight import *

# 注册回调函数
def print_grad(name, grad):
    print('Gradient of', name, ':', grad)


# 超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100 #RNN的隐藏状态下向量的元素个数
time_size = 5 # Truncated BPTT的时间跨度
lr = 0.1
max_epoch = 100

# 读入学习数据（只有前面1000个）
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = torch.from_numpy(corpus[:corpus_size])
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1] # 输入
ts = corpus[1:]  # 输出

data_size = len(xs)
# 单词数量和词汇数量
print(f'corpus size: {corpus_size}, vocabulary size: {vocab_size}')


# 学习用的参数
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []


# 生成模型 
model = SimpleRnnlm(vocab_size=vocab_size,
                    wordvec_size=wordvec_size,
                    hidden_size=hidden_size).cuda()


# embed_W, rnn_Wx, rnn_Wh, affine_W = load_weight('dataset/rnnlm_init').values()
# model.embed.weight = Parameter(torch.from_numpy(embed_W))
# model.rnn.weight_ih_l0 = Parameter(torch.from_numpy(rnn_Wx).T)
# model.rnn.weight_hh_l0 = Parameter(torch.from_numpy(rnn_Wh).T)
# model.linear.weight = Parameter(torch.from_numpy(affine_W).T)


# 1.计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]


for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 2.获取mini-batch
        batch_x = torch.empty((batch_size, time_size), dtype=torch.long).cuda()
        batch_t = torch.empty((batch_size, time_size), dtype=torch.long).cuda()
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 清空梯度，计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.optimizer.zero_grad()
        
        # handles = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         handle = param.register_hook(lambda grad, name=name: print_grad(name, grad))
        #         handles.append(handle)

        # 开始反向传播
        loss.backward()

        # 移除回调函数
        # for handle in handles:
        #     handle.remove()
        
        model.optimizer.step()

        # 重置RNN的网络状态
        model.reset_state()

        total_loss += loss
        loss_count += 1

    # 各个epoch的困惑度评价
    ppl = torch.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f'
          % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 绘制图形
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()