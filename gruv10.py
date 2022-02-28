# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/2/28 15:56

import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


def prepare_data(_batch_size, path='./data'):
    train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor())
    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False)
    return _train_loader, _test_loader


class GRUCell(nn.Module):
    def __init__(self, _in_size, _hid_size):
        super(GRUCell, self).__init__()
        self._in_size = _in_size
        self._hid_size = _hid_size

        self.in2hid = nn.Linear(_in_size, 3 * _hid_size)
        self.hid2hid = nn.Linear(_hid_size, 3 * _hid_size)
        self.adjust_params()

    def adjust_params(self):
        std = 1.0 / math.sqrt(self._hid_size)
        for param in self.parameters():
            param.data.uniform_(-std, std)

    def forward(self, _input, _hid):
        _input = _input.view(-1, _input.size(1))
        input_weighted = self.in2hid(_input)
        hid_weighted = self.hid2hid(_hid)

        input_weighted = input_weighted.squeeze()
        hid_weighted = hid_weighted.squeeze()

        in_r, in_z, in_n = input_weighted.chunk(3, 1)
        hid_r, hid_z, hid_n = hid_weighted.chunk(3, 1)

        reset = torch.sigmoid(in_r + hid_r)
        update = torch.sigmoid(in_z + hid_z)
        new = torch.tanh(in_n + (reset * hid_n))

        hid_updated = new + update * (_hid - new)
        return hid_updated


class GRUModel(nn.Module):
    def __init__(self, _in_size, _hid_size, _out_size):
        super(GRUModel, self).__init__()
        self._in_size = _in_size
        self._hid_size = _hid_size
        self.out_size = _out_size
        self.grucell = GRUCell(_in_size=_in_size, _hid_size=_hid_size)
        self.fc = nn.Linear(_hid_size, _out_size)

    def forward(self, x):
        outputs = []
        if torch.cuda.is_available():
            hid_ori = torch.zeros(x.size(0), self._hid_size).cuda()
        else:
            hid_ori = torch.zeros(x.size(0), self._hid_size)

        hid_updated = hid_ori

        for seq in range(x.size(1)):
            hid_updated = self.grucell(x[:, seq, :], hid_updated)
            outputs.append(hid_updated)

        outs = outputs[-1].squeeze()
        outs = self.fc(outs)
        return outs


def train(_grumodel, _criterion, _optimizer, _train_loader, _seq_len, _epoch_num, _in_size, _test_loader):
    for epoch in range(_epoch_num):
        print(f'Epoch {epoch} start!')
        for images, labels in _train_loader:
            if torch.cuda.is_available():
                images = images.view(-1, _seq_len, _in_size).cuda()
                labels = labels.cuda()
            else:
                images = images.view(-1, _seq_len, _in_size)

            _optimizer.zero_grad()
            outs = _grumodel(images)
            loss = _criterion(outs, labels)
            if torch.cuda.is_available():
                loss.cuda()
            loss.backward()
            _optimizer.step()
        print("Start test!!")
        test(_grumodel, _test_loader, _seq_len, _in_size)


def test(_grumodel, _test_loader, _seq_len, _in_size):
    correct = 0
    total = 0
    for images, labels in _test_loader:
        if torch.cuda.is_available():
            images = images.view(-1, _seq_len, _in_size).cuda()
            labels = labels.cuda()
        else:
            images = images.view(-1, _seq_len, _in_size)

        outs = _grumodel(images)
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
    in_size = 28
    hid_size = 128
    out_size = 10
    epoch_num = 20
    seq_len = 28
    train_loader, test_loader = prepare_data(batch_size)
    grumodel = GRUModel(in_size, hid_size, out_size)
    if torch.cuda.is_available():
        grumodel.cuda()

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(grumodel.parameters(), lr=learning_rate)
    train(grumodel, criterion, optimizer, train_loader, seq_len, epoch_num, in_size, test_loader)

