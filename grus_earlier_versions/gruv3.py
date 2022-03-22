# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import math

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # 这一行暂时不知道该如何处理。


def prepare_data(batch_size, path='./data'):
    train_data = datasets.MNIST(root=path, train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root=path, train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(len(train_data))
    return train_loader, test_loader


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.hidden2hidden = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.reset_premeters()

    def reset_premeters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-std, std)

    def forward(self, input, hidden):
        input = input.view(-1, input.size(1))

        input_weights = self.input2hidden(input)
        hidden_weights = self.hidden2hidden(hidden)

        input_weights = input_weights.squeeze()
        hidden_weights = hidden_weights.squeeze()

        input_weights_reset, input_weights_input, input_weights_new = input_weights.chunk(3, 1)
        hidden_weights_reset, hidden_weights_input, hidden_weights_new = hidden_weights.chunk(3, 1)

        resetgate = torch.sigmoid(input_weights_reset + hidden_weights_reset)
        inputgate = torch.sigmoid(input_weights_input + hidden_weights_input)
        newgate = torch.tanh(input_weights_new + (resetgate * hidden_weights_new))
        hidden_updated = newgate + inputgate * (hidden - newgate)

        return hidden_updated


class GRUCellv2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCellv2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        lower_bound, upper_bound = - math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size)  # 上界和下界
        self.in2hid_w = nn.ParameterList([self.gen_rand_weights(lower_bound, upper_bound, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.gen_rand_weights(lower_bound, upper_bound, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lower_bound, upper_bound, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lower_bound, upper_bound, hidden_size) for _ in range(3)])

    @staticmethod
    def gen_rand_weights(lower_bound, upper_bound, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper_bound - lower_bound) + lower_bound)
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper_bound - lower_bound) + lower_bound)

    def forward(self, input, hid):
        r = torch.sigmoid(torch.mm(input, self.in2hid_w[0]) + self.in2hid_b[0] +
                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(input, self.in2hid_w[1]) + self.in2hid_b[1] +
                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
        n = torch.tanh(torch.mm(input, self.in2hid_w[2]) + self.in2hid_b[2] +
                       torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
        next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
        return next_hid


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = layer_size
        self.output_dim = output_size
        self.grucell = GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_batch):
        if torch.cuda.is_available():
            hidden_batch = Variable(torch.zeros(self.layer_dim, input_batch.size(0), self.hidden_dim).cuda())
        else:
            hidden_batch = Variable(torch.zeros(self.layer_dim, input_batch.size(0), self.hidden_dim))

        # hidden_batch = torch.zeros(input_batch.size(0), self.hidden_size)
        outs = []
        hidden_new = hidden_batch[0, :, :]
        # out = self.grucell(input_batch, hidden_batch)
        # 输入尺寸 batch_size * seq_length * input_size
        for seq in range(input_batch.size(1)):
            hidden_new = self.grucell(input_batch[:, seq, :], hidden_new)
            outs.append(hidden_new)

        out = outs[-1].squeeze()
        out = self.fc(out)
        return out


def train(grumodel, train_loader, test_loader, seq_len, loss_list, iter_num, num_epochs):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_len, input_size).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_len, input_size))
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = grumodel(images)
            loss = criterion(outputs, labels)

            if torch.cuda.is_available():
                loss.cuda()

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            iter_num += 1
            if iter_num % 500 == 0:
                test(test_loader, iter_num, loss)


def train2(seq_len, loss_list, num_epochs, iter_num, train_loader):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_len, input_size).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_len, input_size))
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = grumodel(images)
            loss = criterion(outputs, labels)

            if torch.cuda.is_available():
                loss.cuda()

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            iter_num += 1
            if iter_num % 500 == 0:
                test(test_loader, iter_num, loss)
                # print(loss.item())
                # print('will be tested later...')


def test(test_loader, iter_num, loss):
    correct = 0
    total = 0

    for images, labels in test_loader:
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_len, input_size).cuda())
        else:
            images = Variable(images.view(-1, seq_len, input_size))
        outputs = grumodel(images)

        # torch.max返回两个值，第一个值为最大值，第二个值为最大值的索引，一般计算准确率的时候只关注索引即可。
        # 第二个输入参数如果是0则表示求每列的最大值和最大索引，1则为每行。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        if torch.cuda.is_available():
            correct += (predicted.cuda() == labels.cuda()).sum()
        else:
            correct += (predicted == labels).sum()

        # correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    print(f'Iteration: {iter_num}'.ljust(20),
          f'Loss: {round(loss.item(), 6)}'.ljust(20),
          f'Accuracy: {round(float(accuracy), 6)}'.ljust(20))


if __name__ == '__main__':
    batch_size = 200
    path = '../data'
    train_loader, test_loader = prepare_data(batch_size, path)

    # 指定输入，隐藏和输出尺寸，layer_size指的是中间有几个隐层，这里取一个。seq_len
    input_size = 28
    hidden_size = 128
    layer_size = 1
    output_size = 10
    seq_len = 28

    # 建立模型
    grumodel = GRUModel(input_size, hidden_size, layer_size, output_size)
    if torch.cuda.is_available():
        grumodel.cuda()
    # 指定loss计算方法
    criterion = nn.CrossEntropyLoss()

    # 指定优化方法
    learning_rate = 0.1
    optimizer = torch.optim.SGD(grumodel.parameters(), lr=learning_rate)

    # 指定迭代次数和循环次数
    n_iters = 6000
    num_epochs = 20

    # 设置随机种子，这样每次的结果都是一样的
    torch.manual_seed(125)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(125)

    loss_list = []
    iter_num = 0
    train(grumodel, train_loader, test_loader, seq_len, loss_list, iter_num, num_epochs)
    # train2(seq_len=seq_len,
    #       loss_list=loss_list,
    #       num_epochs=num_epochs,
    #       iter_num=iter_num,
    #       train_loader=train_loader)
