# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/2/28 15:56

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


def prepare_data(_batch_size, path='./data'):
    trans = transforms.Compose([transforms.Resize((56, 56)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=trans)
    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False)
    return _train_loader, _test_loader


class VGG16(nn.Module):
    def __init__(self, in_channels=1, classes=10):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bolck_channels1 = 64 // 8
        self.bolck_channels2 = 128 // 8
        self.bolck_channels3 = 256 // 8
        self.bolck_channels4 = 256 // 8
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.bolck_channels1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            # nn.Conv2d(self.bolck_channels1, self.bolck_channels1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.bolck_channels1, self.bolck_channels2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            # nn.Conv2d(self.bolck_channels2, self.bolck_channels2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.bolck_channels2, self.bolck_channels2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            # nn.Conv2d(self.bolck_channels3, self.bolck_channels3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.Conv2d(self.bolck_channels3, self.bolck_channels3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.bolck_channels2, self.bolck_channels2, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            # nn.Conv2d(self.bolck_channels4, self.bolck_channels4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.Conv2d(self.bolck_channels4, self.bolck_channels4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Conv2d(self.bolck_channels4, self.bolck_channels4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.Conv2d(self.bolck_channels4, self.bolck_channels4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.Conv2d(self.bolck_channels4, self.bolck_channels4, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )
        self.net2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.bolck_channels2 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, self.classes)
        )

    def forward(self, in_put):
        output = self.net(in_put)
        output = self.net2(output)
        return output


def train(_model, _criterion, _optimizer, _train_loader, _epoch_num, _test_loader):
    acc_list = []
    _model.train()
    # for epoch in range(_epoch_num):
    for _ in trange(_epoch_num):
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
    plt.plot(acc_list)
    plt.savefig('accuracy.png')
    plt.show()


def test(_model, _test_loader, ):
    correct = 0
    total = 0
    _model.eval()
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
    batch_size = 256
    out_size = 10
    epoch_num = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_data(batch_size)
    model = VGG16()

    if torch.cuda.is_available():
        model.cuda()
    torchsummary.summary(model, input_size=(1, 56, 56))
    # print(next(model.parameters()).is_cuda)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train(model, criterion, optimizer, train_loader, epoch_num, test_loader)
