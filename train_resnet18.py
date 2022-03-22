# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/3/14 16:11
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
# import torchvision
import torchsummary
from resnet18 import resnet18
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

def prepare_data(_batch_size, path='./data'):
    trans = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(root=path, train=True, transform=trans, download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=trans)
    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False)
    return _train_loader, _test_loader


def train(_model, _criterion, _optimizer, _train_loader, _epoch_num, _test_loader):
    acc_list = []
    _model.train()
    # for epoch in range(_epoch_num):
    with tqdm(total=_epoch_num, desc='Training Epoch', unit='Batch') as pbar:
        for _ in range(_epoch_num):
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
            accuracy = 0
            accuracy = test(_model, _test_loader)
            acc_list.append(accuracy)
            pbar.update()
    print(acc_list)
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

    print(f'Test Accuracy: {round(float(accuracy), 6)}'.ljust(20))
    print('-'*20 + '-' *20 + ' '*20)
    return accuracy

if __name__ == '__main__':
    batch_size = 16
    out_size = 10
    epoch_num = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = prepare_data(batch_size)
    model = resnet18(num_classes=out_size)

    if torch.cuda.is_available():
        model.cuda()
    torchsummary.summary(model, input_size=(1, 224, 224))
    # print(next(model.parameters()).is_cuda)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train(model, criterion, optimizer, train_loader, epoch_num, test_loader)
