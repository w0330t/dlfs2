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

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads


    def predict(self, x):
        pass

    def forward(self, x, t):
        pass

    def backward(self, dout=1):
        pass


if __name__ == '__main__':
    tln = TwoLayerNet(1, 2, 3)
