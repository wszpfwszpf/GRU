# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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

        _in_data_weighted = _in_data_weighted.squeeze()  # 如果没有维度为1的维度，则经过squeeze之后也没有变化。
        _hid_weighted = _hid_weighted.squeeze()

        in_r, in_z, in_n = _in_data_weighted.chunk(3, 1)
        hid_r, hid_z, hid_n = _hid_weighted.chunk(3, 1)

        reset = torch.sigmoid(in_r + hid_r)
        update = torch.sigmoid(in_z + hid_z)
        new = torch.tanh(in_n + (reset * hid_n))
        hid_updated = new + update * (_hid - new)
        return hid_updated


class GRUModel(nn.Module):
    def __init__(self, _in_size, _hid_size, _out_size):
        super(GRUModel, self).__init__()
        self.hid_size = _hid_size
        self.gru_cell = GRUCell(_in_size, _hid_size)
        self.fc = nn.Linear(_hid_size, _out_size)

    def forward(self, _in_data):
        outputs = []
        if torch.cuda.is_available():
            hid_ori = torch.zeros(_in_data.size(0), self.hid_size).cuda()
        else:
            hid_ori = torch.zeros(_in_data.size(0), self.hid_size)
        hid_updated = hid_ori
        # _in_data的第二个维度对应seq_len
        for seq in range(_in_data.size(1)):
            hid_updated = self.gru_cell(_in_data[:, seq, :], hid_updated)
            outputs.append(hid_updated)

        outs = outputs[-1].squeeze()
        outs = self.fc(outs)
        return outs


def train(_model, _train_loader, _criterion, _optimizer, _epoch_nums, _test_loader, _seq_len, _in_size):
    # flags = 1
    for epoch in range(_epoch_nums):
        print(f'Epoch: {epoch}')
        for images, labels in _train_loader:
            # if flags:
            #     print(images.size())
            #     flags = 0
            if torch.cuda.is_available():
                # images的格式为torch.Size([200, 1, 28, 28])，要把它变成200*28*28的，因此要用view。
                images = images.view(-1, _seq_len, _in_size).cuda()
                labels = labels.cuda()   # 必须要这么些，用下面这一句会提示出错。
                # labels.cuda()
            else:
                images = images.view(-1, _seq_len, _in_size)

            _optimizer.zero_grad()
            outs = _model(images)
            loss = _criterion(outs, labels)
            if torch.cuda.is_available():
                loss.cuda()
            loss.backward()
            _optimizer.step()
        print("Start test")
        test(_model, _test_loader, _seq_len, _in_size)


def test(_model, _test_loader, _seq_len, _in_size):
    correct = 0
    total = 0
    for images, labels in _test_loader:
        if torch.cuda.is_available():
            images = images.view(-1, _seq_len, _in_size).cuda()
            # labels = labels.cuda()

        outs = _model(images)
        _, predicted = torch.max(outs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cuda() == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Accuracy: {round(float(accuracy), 6)}'.ljust(20))


if __name__ == '__main__':
    # 设置随机种子，这样每次的结果都是一样的
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    in_size = 28
    hid_size = 128
    out_size = 10
    seq_len = 28
    epochs_num = 2
    batch_size = 200

    train_loader, test_loader = prepare_data(batch_size)

    model = GRUModel(in_size, hid_size, out_size)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    lr = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train(model, train_loader, criterion, optimizer, epochs_num, test_loader, seq_len, in_size)
