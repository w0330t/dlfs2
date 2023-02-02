import torch

# layers
class MatMul:
    def __init__(self, W, b):
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
        return dx

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [torch.zeros_like(W), torch.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = torch.mm(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = torch.mm(dout, W.T)
        dW = torch.mm(self.x.T, dout)
        db = torch.sum(dout, dim = 0)
        self.grads[0] = dW.clone()
        self.grads[1] = db.clone()
        return dx
    

# =============Activate=============

class Sorfmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.x = None

    def forward(self, x):
        if x.ndim == 2:
            c = x.max(dim=1)
            x_exp = torch.exp(x - c.values.unsqueeze(0).t())
            self.x = x_exp / x_exp.sum(dim=1).unsqueeze(0).t()
            return self.x
        elif x.ndim == 1:
            t = t.reshape(1, len(t))
            x = x.reshape(1, len(x))
            c = x.max()
            x_exp = torch.exp(x - c)
            self.x = x_exp / x_exp.sum()
            return self.x

    def backward(self, dout):
        """反向传播
            dout (torch.Tensor): 这里传入的其实是标签
        """
        dx = self.x - dout
        return dx


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + torch.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


# ===============Loss=============
class CrossEntropy:
    def __init__(self) -> None:
        self.out = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        if x.ndim == 1:
            # 如果维度为1，直接格式化为一行 \
            t = t.reshape(1, len(t))
            x = x.reshape(1, len(x))

        # 取行值 \
        batch_size = t.shape[0]
        self.out = -torch.sum(t * torch.log(x + 1e-7)) / batch_size
        return self.out

    def backward(self):
        return self.t



# test
if __name__ == '__main__':
    # 创建权重W
    W = torch.tensor([[1., -1.], [1., -1.]], requires_grad=True)
    # 创建输入矩阵x
    x = torch.tensor([[1., 2.], [3., 4.]])
    # 创建偏置
    b = torch.tensor(2.)
    # 创建标签
    t = torch.tensor([[0, 1], [1, 0]])

    # 创建MatMul对象
    test = Sorfmax()
    # 前向传播
    forward_out = test.forward(x)
    print("Forward Out:", forward_out)

    # 反向传播
    dout = torch.ones_like(forward_out)
    backward_out = test.backward(dout)
    print("Backward Out:", backward_out)
