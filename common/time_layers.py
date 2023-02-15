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

        self.cache = (x, h_prev, h_next)
        return h_next