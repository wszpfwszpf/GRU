# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def prepare_data(_batch_size, path='./data'):
    train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor())

    _train_loader = DataLoader(train_data, _batch_size, shuffle=True)
    _test_loader = DataLoader(test_data, _batch_size, shuffle=False)
    return _train_loader, _test_loader


class GRUCell(nn.Module):
    def __init__(self, _in_size, _hid_size):
        super(GRUCell, self).__init__()
        self.in_size = _in_size
        self.hid_size = _hid_size
        self.in2hid = nn.Linear(_in_size, 3 * _hid_size, bias=True)
        self.hid2hid = nn.Linear(_hid_size, 3 * _hid_size, bias=True)
        self.adjust_params()

    def adjust_params(self):
        std = 1.0 / math.sqrt(self.hid_size)
        for param in self.parameters():
            # param.data.uniform_(-std, std)
            param.data.uniform_(-std, std)

    def forward(self, x, hid):  # x 一维二维均可
        x = x.view(-1, x.size(1))
        gate_x = self.in2hid(x)
        gate_hid = self.hid2hid(hid)

        gate_x = gate_x.squeeze()
        gate_hid = gate_hid.squeeze()

        x_reset, x_x, x_new = gate_x.chunk(3, 1)
        hid_reset, hid_x, hid_new = gate_hid.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + hid_reset)
        input_gate = torch.sigmoid(x_x + hid_x)
        new_gate = torch.tanh(x_new + (reset_gate * hid_new))

        hid_updated = new_gate + input_gate * (hid - new_gate)

        return hid_updated


class GRUModel(nn.Module):
    def __init__(self, _in_size, _hid_size, _out_size):
        super(GRUModel, self).__init__()
        self.in_size = _in_size
        self.hid_size = _hid_size
        self.out_size = _out_size
        self.grucell = GRUCell(_in_size, _hid_size)
        self.fc = nn.Linear(_hid_size, _out_size)

    def forward(self, in_data):  # 接受的应该是batch_size * seq_len * in_dim的数据
        outs = []
        if torch.cuda.is_available():
            hid = Variable(torch.zeros(in_data.size(0), self.hid_size).cuda())
        else:
            hid = Variable(torch.zeros(in_data.size(0), self.hid_size))
        # hid = torch.zeros(self.in_data.size(0), self.hid_size)
        hid_updated = hid
        for seq in range(in_data.size(1)):
            hid_updated = self.grucell(in_data[:, seq, :], hid_updated)
            outs.append(hid_updated)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


def train():
    for epoch in range(epoch_nums):
        print('-' * 20 + f'Epoch: {epoch} start' + '-' * 20)
        for images, labels in train_loader:
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_len, in_size).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_len, in_size))
                labels = Variable(labels)

            optimizer.zero_grad()  # 清空过往梯度
            outs = grumodel(images)
            # correct += (outs == labels)

            loss = criterion(outs, labels)   # 输入图像和标签，通过infer计算得到预测值，计算损失函数
            if torch.cuda.is_available():
                loss.cuda()
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度更新网络参数
        print('-' * 20 + f'Epoch: {epoch} ended' + '-' * 20)
        test()


def test():
    print('-' * 20 + 'Test start!' + '-' * 24)
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_len, in_size).cuda())
        else:
            images = Variable(images.view(-1, seq_len, in_size))

        outs = grumodel(images)

        _, predicted = torch.max(outs.data, 1)
        # correct += (outs == labels)
        total += labels.size(0)

        if torch.cuda.is_available():
            correct += (predicted.cuda() == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Accuracy: {round(float(accuracy), 6)}'.ljust(20))


if __name__ == '__main__':
    batch_size = 200
    in_size = 28
    seq_len = 28
    hid_size = 128
    out_size = 10
    epoch_nums = 20
    iter_nums = 6000  # 好像没什么用
    learning_rate = 0.1

    # 设置随机种子，这样每次的结果都是一样的
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    grumodel = GRUModel(in_size, hid_size, out_size)
    if torch.cuda.is_available():
        grumodel.cuda()

    train_loader, test_loader = prepare_data(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(grumodel.parameters(), lr=learning_rate)
    train()
