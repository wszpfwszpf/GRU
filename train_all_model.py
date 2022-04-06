# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/3/25 9:30

import torch
# import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import torchvision
import time
import copy
# import numpy as np
from torchsummary import torchsummary
from Inceptions.InceptionV1 import InceptionV1
from torch.utils.data import DataLoader
from Densenets.Denset_efficient import DenseNet
from DPN.dpn import dpn92

def prepare_data(_batch_size, path='data/cifar10-data'):
    # trans = transforms.Compose([transforms.Resize((224, 224)),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize((0.1307,), (0.3081,))])
    trans = torchvision.transforms.Compose([
        # torchvision.transforms.Pad(4),
        # torchvision.transforms.RandomCrop((32, 32)),
        # torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #ImageNet
    ])
    train_data = datasets.CIFAR10(root=path, train=True, transform=trans, download=True)
    test_data = datasets.CIFAR10(root=path, train=False, transform=trans)
    _train_loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True, num_workers=4)
    _test_loader = DataLoader(test_data, batch_size=_batch_size, shuffle=False, num_workers=4)
    dataloaders_dict = {'train': _train_loader, 'test': _test_loader}

    return dataloaders_dict


def train_and_test(model, dataloaders_dict, _criterion, optimizer, num_epochs, device):
    since = time.time()
    phases = ['train', 'test']
    acc_history = {
        'train': [],
        'test': [],
    }
    loss_history = {
        'train': [],
        'test': [],
    }
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # for epoch in trange(total=num_epochs, desc='Train and Test', unit='Epoch'):
    for epoch in trange(num_epochs, desc='Train and Test', unit='Epoch'):
        # for epoch in range(num_epochs):
        #     print(f'Epoch:{epoch}')
        for phase in phases:
            # print(f'Phase:{phase}')
            if phase == 'train':
                model.train()
                # print('model.train()')
            elif phase == 'test':
                model.eval()
                # print('model.eval()')

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    outs = model(inputs)
                    loss = _criterion(outs, labels)
                    _, preds = torch.max(outs.data, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels).item()
            acc = running_corrects / len(dataloaders_dict[phase])
            print('acc', acc)
            loss = running_loss / len(dataloaders_dict[phase])

            acc_history[phase].append(acc)
            loss_history[phase].append(loss)
            if phase == 'test':
                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, acc_history, loss_history, best_acc

def train(_model, _criterion, _optimizer, _train_loader, _epoch_num, _test_loader):
    since = time.time()
    # test_acc_history = []
    # train_acc_history = []
    # train_loss_history = []
    # test_loss_history = []
    # best_model_wts = copy.deepcopy(model.state_dict())
    train_acc_list = []
    test_acc_list = []
    best_acc = 0.0

    _model.train()
    # with tqdm(total=_epoch_num, desc='Training Epoch', unit='Batch') as pbar:
    # for epoch in trange(_epoch_num, desc='Training Epoch', unit='Batch'):
    for epoch in range(_epoch_num):
        print(f'Epoch {epoch + 1} start!')
        if epoch == 20:
            optimizer.param_groups[0]['lr'] /= 10
        if epoch == 40:
            optimizer.param_groups[0]['lr'] /= 10

        train_running_loss = 0.0
        total = 0
        correct = 0
        with tqdm(total=len(_train_loader), desc='Training Epoch', unit='Batch') as pbar:
            for images, labels in _train_loader:
                # if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

                _optimizer.zero_grad()
                outs = _model(images)
                # outs = _model(images)
                _, predicted = torch.max(outs.data, 1)
                loss = _criterion(outs, labels)
                # if torch.cuda.is_available():
                correct += (predicted.cuda() == labels.cuda()).sum()
                total += labels.size(0)
                # else:
                #     correct += (predicted == labels).sum()
                train_running_loss += loss.item() * images.size(0)
                # if torch.cuda.is_available():
                loss.cuda()
                loss.backward()
                _optimizer.step()
                pbar.update()
        # print("Start test!!")
        # train_epoch_loss = train_running_loss / len(_train_loader)
        # train_loss_history.append(train_epoch_loss)
        # train_acc_history.append(correct / len(_train_loader))

        accuracy, test_epoch_loss = test(_model, _test_loader, _criterion)
        print(f'Training Accuracy:{100 * correct / total}')
        train_acc_list.append(100 * correct / total)
        print(f'Test Accuracy: {round(float(accuracy), 6)}')
        test_acc_list.append(accuracy)
        # test_loss_history.append(test_epoch_loss)
        # test_acc_history.append(accuracy)

    time_elapsed = time.time() - since
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    print('Training complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # model.load_state_dict(best_model_wts)
    # return model, test_acc_history, train_acc_history, best_acc


def test(_model, _test_loader, _criterion):
    correct = 0
    total = 0
    test_running_loss = 0.0

    _model.eval()
    with torch.no_grad():
        with tqdm(total=len(_test_loader), desc='Training Epoch', unit='Batch') as pbar:
            for images, labels in _test_loader:
                images = images.cuda()
                labels = labels.cuda()
                outs = _model(images)
                loss = _criterion(outs, labels)
                _, predicted = torch.max(outs.data, 1)
                total += labels.size(0)
                test_running_loss += loss.item() * images.size(0)
                if torch.cuda.is_available():
                    correct += (predicted.cuda() == labels.cuda()).sum()
                else:
                    correct += (predicted == labels).sum()
                pbar.update()
    accuracy = 100 * correct / total
    test_epoch_loss = test_running_loss / total
    # print(f'Test Accuracy: {round(float(accuracy), 6)}'.ljust(20))
    # print('-' * 20 + '-' * 20 + ' ' * 20)
    return accuracy, test_epoch_loss

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    batch_size = 32
    lr = 0.01
    weight_decay = 0.0005
    num_epochs = 40
    # model = InceptionV1(num_classes=10)
    # model = DenseNet()
    model = dpn92(10)
    # model = InceptionNet(num_classes=10)
    model.to(device)
    # torchsummary.summary(model, input_size=(3, 32, 32))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    dataloaders_dict = prepare_data(_batch_size=batch_size)
    # model, acc_history, loss_history, best_acc = train_and_test(model, dataloaders_dict, criterion, optimizer,
    #                                                             num_epochs, device)
    # print('acc_history:', acc_history)
    # print('loss_history', loss_history)
    # print('best_acc', best_acc)
    # savehis(acc_history['test'], acc_history['train'],'record', )
    train(model, criterion, optimizer, dataloaders_dict['train'], num_epochs, dataloaders_dict['test'])
