import torch

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [torch.zeros_like(Wx), 
                        torch.zeros_like(Wh),
                        torch.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = torch.dot(h_prev, Wh) + torch.dot(x, Wx) + b
        h_next = torch.tanh(t)

        # 留下缓存供反向传播使用
        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dtanh = dh_next * (1 - h_next ** 2)
        db = torch.sum(dtanh, dim=0)
        dWh = torch.dot(h_prev.t(), dtanh)
        dh_prev = torch.dot(dtanh, Wh.T)
        dWx = torch.dot(x.T, dtanh)
        dx = torch.dot(dtanh, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [torch.zeros_like(Wx),
                      torch.zeros_like(Wh),
                      torch.zeros_like(b)]
        # 用于保存多个RNN层
        self.layers = None

        # h 用于保存向前传播的时候最后一个层的状态
        # dh 用于保存前一个块的隐藏梯度
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = torch.empty((N, T, H))

        if not self.stateful or self.h is None:
            self.h = torch.zeros((N, H))

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs