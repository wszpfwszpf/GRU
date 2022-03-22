# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torch.nn import Parameter
# from torch import Tensor
# import torch.nn.functional as F
import math

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(125)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(125)


# 下载数据
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

batch_size = 200
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
print(len(train_dataset))
print(num_epochs)

# 加载数据
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
# 测试数据不需要打乱
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


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

        #         rt = F.sigmoid(w_xt_r * xt + w_ht_1_r * ht_1)
        #         zt = F.sigmoid(w_xt_z * xt + w_ht_1_z * ht_1)
        #         ht_hat = F.tanh(w_xt_h * xt + rt * ht_1)
        #         out = (1-zt) * ht_1 + zt * ht_hat

        return hy


class GRUCellv2(nn.Module):
    """自定义GRUCell"""
    def __init__(self, input_size, hidden_size):
        super(GRUCellv2, self).__init__()
        # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置)
        # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
        lb, ub = - math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

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
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self.gru_cellv2 = GRUCellv2(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        outs = []
        hn = h0[0, :, :]
        hnv2 = hn

        for seq in range(x.size(1)):
            # hn = self.gru_cell(x[:,seq,:], hn)
            hnv2 = self.gru_cellv2(x[:, seq, :], hnv2)
            # outs.append(hn)
            outs.append(hnv2)

        out = outs[-1].squeeze()
        out = self.fc(out)
        # out.size() --> 100, 10
        return out


input_dim = 28
hidden_dim = 128
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 10

# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)

#######################
#  USE GPU FOR MODEL  #
#######################

if torch.cuda.is_available():
    model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train(seq_dim, loss_list, num_epochs, iter_num, train_loader):
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.view(-1, seq_dim, input_dim).cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images.view(-1, seq_dim, input_dim))
                labels = Variable(labels)

            optimizer.zero_grad()

            # Forward pass to get output/logits
            # outputs.size() --> 100, 10
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            if torch.cuda.is_available():
                loss.cuda()

            # Getting gradients w.r.t. parameters
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            iter_num += 1
            if iter_num % 500 == 0:
                #                 print('will be tested later...')
                test(iter_num, loss)


def test(iter_num, loss):
    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in test_loader:
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            images = Variable(images.view(-1, seq_dim, input_dim).cuda())
        else:
            images = Variable(images.view(-1, seq_dim, input_dim))

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        if torch.cuda.is_available():
            correct += (predicted.cpu() == labels.cpu()).sum()
            # if predicted.cpu() == labels.cpu():
            #     correct += 1
            # # correct += (predicted.cpu() == labels.cpu())
        else:
            correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    # Print Loss
    print(f'Iteration: {iter_num}'.ljust(20),
          f'Loss: {round(loss.item(),6)}'.ljust(20),
          f'Accuracy: {round(float(accuracy),6)}'.ljust(20))


'''
STEP 7: TRAIN THE MODEL
'''
# Number of steps to unroll
seq_dim = 28
loss_list = []
iter_num = 0
flag = 1
num_epochs = 10
train(seq_dim, loss_list, num_epochs, iter_num, train_loader)
