# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def prepare_data(_batch_size, _path='./data'):
    train_data = datasets.MNIST(root=_path, train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root=_path, train=False, transform=transforms.ToTensor())

    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False)
    return _train_loader, _test_loader


class GRUCell(nn.Module):
    def __init__(self, _in_size, _hid_size):
        super(GRUCell, self).__init__()
        self._in_size = _in_size
        self._hid_size = _hid_size
        self.in2hid = nn.Linear(_in_size, 3 * _hid_size, bias=True)
        self.hid2hid = nn.Linear(_hid_size, 3 * _hid_size, bias=True)
        self.adjust_params()

    def adjust_params(self):
        std = 1.0 / math.sqrt(self._hid_size)
        for param in self.parameters():
            param.data.uniform_(-std, std)

    def forward(self, _in_data, _hid):
        _in_data = _in_data.view(-1, _in_data.size(1))

        _in_data_weighted = self.in2hid(_in_data)
        _hid_weighted = self.hid2hid(_hid)

        # _in_data_weighted = _in_data_weighted.squeeze()  # 如果没有维度为1的维度，则经过squeeze之后也没有变化。
        # _hid_weighted = _hid_weighted.squeeze()

        _in_r, _in_z, _in_n = _in_data_weighted.chunk(3, 1)
        _hid_r, _hid_z, _hid_n = _hid_weighted.chunk(3, 1)

        reset = torch.sigmoid(_in_r + _hid_r)
        ingate = torch.sigmoid(_in_z + _hid_z)
        new = torch.tanh(_in_n + (reset * _hid_n))
        hid_updated = new + ingate * (_hid - new)

        return hid_updated


class GRUModel(nn.Module):
    def __init__(self, _in_size, _hid_size, _out_size):
        super(GRUModel, self).__init__()
        self._in_size = _in_size
        self._hid_size = _hid_size
        self._out_size = _out_size
        self.grucell = GRUCell(_in_size, _hid_size)
        self.fc = nn.Linear(_hid_size, _out_size)

    def forward(self, _in_data):
        if torch.cuda.is_available():
            _hid_ori = Variable(torch.zeros(_in_data.size(0), self._hid_size).cuda())
        else:
            _hid_ori = Variable(torch.zeros(_in_data.size(0), self._hid_size))
        outs = []
        _hid_updated = _hid_ori
        for seq in range(_in_data.size(1)):
            _hid_updated = self.grucell(_in_data[:, seq, :], _hid_updated)
            outs.append(_hid_updated)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


def train(_grumodel, _criterion, _optimizer, _train_loader, _test_loader, _epoch_nums):
    for epoch in range(_epoch_nums):
        print('Epoch:', epoch)
        for images, labels in _train_loader:
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_len, in_size).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_len, in_size))
                labels = Variable(labels)

            _optimizer.zero_grad()

            outs = _grumodel(images)
            loss = _criterion(outs, labels)
            if torch.cuda.is_available():
                loss.cuda()
            loss.backward()
            _optimizer.step()
        print("Start test")
        test(_test_loader)


def test(_test_loader):
    correct = 0
    total = 0
    for images, labels in _test_loader:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_len, in_size).cuda())
        else:
            images = Variable(images.view(-1, seq_len, in_size))

        outs = grumodel(images)

        _, predicted = torch.max(outs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cuda() == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Accuracy: {round(float(accuracy), 6)}'.ljust(20))


if __name__ == '__main__':
    batch_size = 200
    train_loader, test_loader = prepare_data(batch_size)

    in_size = 28
    hid_size = 128
    out_size = 10
    seq_len = 28
    epochs_nums = 20

    grumodel = GRUModel(in_size, hid_size, out_size)
    if torch.cuda.is_available():
        grumodel.cuda()

    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    optimizer = torch.optim.SGD(grumodel.parameters(), lr=lr)

    # 设置随机种子，这样每次的结果都是一样的
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)
    train(grumodel, criterion, optimizer, train_loader, test_loader, epochs_nums)
