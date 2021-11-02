# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def prepare_data(batch_size, path='./data'):
    # 下载数据
    train_dataset = datasets.MNIST(root=path,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root=path,
                                  train=False,
                                  transform=transforms.ToTensor())

    # 加载数据
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    # 测试数据不需要打乱
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)  # 为什么是乘以三倍隐藏层size呢？ 和后面的chunk函数相对应。
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

        # 调整偏差，感觉和normalization的功能一致

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    # 这块本质上也是公式，但是是用nn.linear的形式，因为nn.linear在生成的时候已经把权重和偏置计算在内了，因此下面可以直接用结果，就是i_r,h_r等六个参数
    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))  # 变成二维的张量。

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)  # 沿1轴分成三块，也就是沿y轴，竖着切。
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUCellv2(nn.Module):
    """
    和GRUCell是一个东西，写法不一致，内部本质上是一致的。
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCellv2, self).__init__()
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置)
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
        lb, ub = - math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

    # 随机初始化
    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    # 这个是完全根据gru的公式来的。
    def forward(self, x, hid):
        r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
                       torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
        next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
        return next_hid


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.grucell = GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            hidden = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            hidden = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        hn = hidden[0, :, :]
        for seq in range(x.size(1)):
            hn = self.grucell(x[:, seq, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out





def train(seq_len, loss_list, num_epochs, iter_num, train_loader):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_len, input_dim).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_len, input_dim))
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if torch.cuda.is_available():
                loss.cuda()

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            iter_num += 1
            if iter_num % 500 == 0:
                test(iter_num, loss)
                # print(loss.item())
                # print('will be tested later...')


def test(iter_num, loss):
    correct = 0
    total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_len, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_len, input_dim).cuda())

        outputs = model(images)

        # correct += (outs == labels)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        if torch.cuda.is_available():
            correct += (predicted.cuda() == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Iteration: {iter_num}'.ljust(20),
          f'Loss: {round(loss.item(), 6)}'.ljust(20),
          f'Accuracy: {round(float(accuracy), 6)}'.ljust(20))


if __name__ == '__main__':
    input_dim = 28
    hidden_dim = 128
    layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
    output_dim = 10

    # model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    path = './data'  # 存放数据的位置
    batch_size = 200
    train_loader, test_loader = prepare_data(batch_size, path)
    n_iters = 6000
    num_epochs = 20
    # num_epochs = n_iters / (len(train_dataset) / batch_size)
    # num_epochs = int(num_epochs)
    # print(len(train_dataset))
    # print(num_epochs)

    # 设置随机种子，这样每次的结果都是一样的
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    seq_len = 28
    loss_list = []
    iter_num = 0
    train(seq_len=seq_len,
          loss_list=loss_list,
          num_epochs=num_epochs,
          iter_num=iter_num,
          train_loader=train_loader)

