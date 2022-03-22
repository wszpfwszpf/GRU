# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/3/15 9:29


# import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
# import torchvision
import torchsummary
import matplotlib.pyplot as plt
from tqdm import trange
import time


def prepare_data(_batch_size, path='./data'):
    trans = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=trans)
    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return _train_loader, _test_loader


class Lenet(nn.Module):
    def __init__(self, in_channels=1, classes=10):
        super(Lenet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, in_put):
        out = self.conv1(in_put)
        out = self.conv2(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def train(_model, _criterion, _optimizer, _train_loader, _epoch_num, _test_loader):
    starttime = time.time()
    acc_list = []
    _model.train()
    # for epoch in range(_epoch_num):
    for _ in trange(_epoch_num, desc='Lenet Training', unit='Epoch'):
        # print(f'Epoch {epoch + 1} start!')
        for images, labels in _train_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            _optimizer.zero_grad()
            outs = _model(images)
            loss = _criterion(outs, labels)
            if torch.cuda.is_available():
                loss.cuda()
            loss.backward()
            _optimizer.step()
        # print("Start test!!")
        accuracy = test(_model, _test_loader)
        acc_list.append(accuracy)
    # print(acc_list.)
    end = time.time() - starttime

    print(end)
    plt.plot(acc_list)
    plt.savefig('accuracy.png')
    plt.show()


def test(_model, _test_loader, ):
    correct = 0
    total = 0
    _model.eval()
    with torch.no_grad():
        for images, labels in _test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outs = _model(images)
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (predicted.cuda() == labels.cuda()).sum()
            else:
                correct += (predicted == labels).sum()
    accuracy = 100 * correct / total

    # print(f'Test Accuracy: {round(float(accuracy), 6)}'.ljust(20))
    # print('-'*20 + '-' *20 + ' '*20)
    return accuracy


if __name__ == '__main__':
    batch_size = 64
    out_size = 10
    epoch_num = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_data(batch_size)
    model = Lenet()

    if torch.cuda.is_available():
        model.cuda()
    torchsummary.summary(model, input_size=(1, 32, 32))
    # print(next(model.parameters()).is_cuda)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train(model, criterion, optimizer, train_loader, epoch_num, test_loader)
