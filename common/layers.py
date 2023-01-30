import torch

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [torch.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        self.x = x
        out = torch.mm(x, W)
        return out

    def backward(self, dout):
        W, = self.params
        dx = torch.mm(dout, W.T)
        dW = torch.mm(self.x.T, dout)
        self.grads[0] = dW.clone()
        return dW

if __name__ == '__main__':
    # 创建权重W
    W = torch.tensor([[1., -1.], [1., -1.]], requires_grad=True)
    # 创建输入矩阵x
    x = torch.tensor([[1., 2.], [3., 4.]])

    # 创建MatMul对象
    matmul = MatMul(W)
    # 前向传播
    out = matmul.forward(x)
    print("out:", out)

    # 反向传播
    dout = torch.ones_like(out)
    matmul.backward(dout)
    print("W.grad:", matmul.grads)
