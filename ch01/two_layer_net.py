import sys
sys.path.append('.')
import numpy as np
import torch
from common.layers import Affine, Sigmoid, Sorfmax, CrossEntropy

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:

        # 初始化权重和偏置
        W1 = 0.01 * torch.randn(input_size, hidden_size)
        b1 = torch.zeros(hidden_size)
        W2 = 0.01 * torch.randn(hidden_size, output_size)
        b2 = torch.zeros(output_size)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
            Sorfmax()
        ]

        self.loss_layer = CrossEntropy()

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            # print(self.params)
            # print(self.grads)


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


if __name__ == '__main__':
    tln = TwoLayerNet(1, 2, 3)
