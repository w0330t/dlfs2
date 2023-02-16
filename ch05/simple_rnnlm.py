import torch
import torch.nn as nn

class SimpleRnnlm(nn.Module):
    def __init__(self, vocab_size, wordvec_size, hidden_size, learning_rate=0.1):
        super(SimpleRnnlm, self).__init__()
        self.hidden_size = hidden_size

        # 初始化词嵌入层
        self.embed = nn.Embedding(vocab_size, wordvec_size)
        self.rnn = nn.RNN(wordvec_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)

        # 记录参数和梯度
        self.params, self.grads = [], []
        # 把所有的参数和迭代器遍历一遍，没有初始化的添加初始化，然后将参数全部存到上面两个值取
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            self.params.append(param)
            self.grads.append(param.grad)

        self.optimiser = torch.optim.SGD(self.parameters(),lr=learning_rate)

    def forward(self, xs, ts):
        xs = self.embed(xs)
        hs, _ = self.rnn(xs)
        ys = self.softmax(self.out(hs))
        loss = torch.mean(-torch.log(torch.gather(ys, 2, ts.view(-1, 1, 1))))
        return loss

    def reset_state(self):
        self.rnn.flatten_parameters()