# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/3/17 19:34
import torch
import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision
import torchsummary
# import matplotlib.pyplot as plt
# from tqdm import trange, tqdm

channel = [
    [192, 64, 96, 128, 16, 32, 32],  # 3a
    [256, 128, 128, 192, 32, 96, 64],  # 3b
    [480, 192, 96, 208, 16, 48, 64],  # 4a
    [512, 160, 112, 224, 24, 64, 64],  # 4b
    [512, 128, 128, 256, 24, 64, 64],  # 4c
    [512, 112, 144, 288, 32, 64, 64],  # 4d
    [528, 256, 160, 320, 32, 128, 128],  # 4e
    [832, 256, 160, 320, 32, 128, 128],  # 5a
    [832, 384, 192, 384, 48, 128, 128]  # 5b
]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Inception(nn.Module):
    def __init__(self, channel, batch_norm=False):
        super(Inception, self).__init__()
        if batch_norm is False:
            self.branch1x1 = nn.Conv2d(channel[0], channel[1], kernel_size=(1, 1), stride=(1, 1))
            self.branch3x3_1 = nn.Conv2d(channel[0], channel[2], kernel_size=(1, 1), stride=(1, 1))
            self.branch3x3_2 = nn.Conv2d(channel[2], channel[3], kernel_size=(3, 3), stride=(1, 1), padding=1)

            self.branch5x5_1 = nn.Conv2d(channel[0], channel[4], kernel_size=(1, 1), stride=(1, 1))
            self.branch5x5_2 = nn.Conv2d(channel[4], channel[5], kernel_size=(5, 5), stride=(1, 1), padding=2)
            self.branchM_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.branchM_2 = nn.Conv2d(channel[0], channel[6], kernel_size=(1, 1), stride=(1, 1))
        else:
            self.branch1x1 = BasicConv2d(channel[0], channel[1], kernel_size=(1, 1), stride=(1, 1))
            self.branch3x3_1 = BasicConv2d(channel[0], channel[2], kernel_size=(1, 1), stride=(1, 1))
            self.branch3x3_2 = BasicConv2d(channel[2], channel[3], kernel_size=(3, 3), stride=(1, 1), padding=1)

            self.branch5x5_1 = BasicConv2d(channel[0], channel[4], kernel_size=(1, 1), stride=(1, 1))
            self.branch5x5_2 = BasicConv2d(channel[4], channel[5], kernel_size=(5, 5), stride=(1, 1), padding=2)

            self.branchM_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.branchM_2 = BasicConv2d(channel[0], channel[6], kernel_size=(1, 1), stride=(1, 1))

        self.relu = nn.ReLU(True)

    def forward(self, x):
        branch1x1 = self.relu(self.branch1x1(x))

        branch3x3_1 = self.relu(self.branch3x3_1(x))
        branch3x3_2 = self.relu(self.branch3x3_2(branch3x3_1))

        branch5x5_1 = self.relu(self.branch5x5_1(x))
        branch5x5_2 = self.relu(self.branch5x5_2(branch5x5_1))

        branch_m_1 = self.relu(self.branchM_1(x))
        branch_m_2 = self.relu(self.branchM_2(branch_m_1))

        outputs = [branch1x1, branch3x3_2, branch5x5_2, branch_m_2]

        return torch.cat(outputs, 1)


class InceptionNet(nn.Module):
    def __init__(self, num_classes=1000, batch_norm=False):
        super(InceptionNet, self).__init__()
        if num_classes == 10:
            channel[0][0] = 64
            self.begin = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(True),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
                nn.ReLU(True)
            )
            self.auxout1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(3, 3)),
                nn.ReLU(True),
                nn.Conv2d(512, 128, kernel_size=(1, 1)),  # 4x4x128
                nn.ReLU(True),
                nn.Conv2d(128, 10, kernel_size=(4, 4))  # 1x1x10
            )
            self.auxout2 = nn.Sequential(
                nn.Conv2d(528, 528, kernel_size=(5, 5), stride=(3, 3)),  # 4x4x528,
                nn.ReLU(True),
                nn.Conv2d(528, 128, kernel_size=(1, 1)),  # 4x4x128,
                nn.ReLU(True),
                nn.Conv2d(128, 10, kernel_size=(4, 4))  # 1x1x10
            )
        else:
            self.begin = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.auxout1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(5, 5), stride=(3, 3)),  # 4x4x512
                nn.ReLU(True),
                nn.Conv2d(512, 128, kernel_size=(1, 1)),  # 4x4x128
                nn.ReLU(True)
            )
            self.auxout12 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
            self.auxout2 = nn.Sequential(
                nn.Conv2d(528, 528, kernel_size=(5, 5), stride=(3, 3)),  # 4x4x528
                nn.ReLU(True),
                nn.Conv2d(528, 128, kernel_size=(1, 1)),  # 4x4x128
                nn.ReLU(True)
            )
            self.auxout22 = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(channel[0], batch_norm)
        self.inception3b = Inception(channel[1], batch_norm)
        self.inception4a = Inception(channel[2], batch_norm)
        self.inception4b = Inception(channel[3], batch_norm)
        self.inception4c = Inception(channel[4], batch_norm)
        self.inception4d = Inception(channel[5], batch_norm)
        self.inception4e = Inception(channel[6], batch_norm)

        self.inception5a = Inception(channel[7], batch_norm)
        self.inception5b = Inception(channel[8], batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1 = nn.Conv2d(1024, num_classes, kernel_size=(1, 1))

        self._initialize_weights()

    '''
    #follow the original papar,but for the computation ,I do not use it
    self.drop=nn.Dropout()
    self.linear=nn.Linear(1024,1000)
    '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.begin(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        # auxout1 = self.auxout1(x)
        # auxout1 = auxout1.view(auxout1.size()[0], -1)
        # if you use this network to train on ImageNet you should add this code
        # auxout1=self.auxout12(auxout1)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # auxout2 = self.auxout2(x)
        # auxout2 = auxout2.view(auxout2.size(0), -1)
        # if you use this network to train on ImageNet you should add this code
        # auxout2=self.auxout22(auxout2)
        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        outputs = self.conv1x1(x)
        outputs = outputs.view(outputs.size(0), -1)

        # return outputs, auxout1, auxout2

        return outputs


# 下面是自己搭的，按照论文中的结构，没有辅助输出。
inception_layer_config = {
    '3a': [192, 64, 96, 128, 16, 32, 32],
    '3b': [256, 128, 128, 192, 32, 96, 64],
    '4a': [480, 192, 96, 208, 16, 48, 64],  # 4a
    '4b': [512, 160, 112, 224, 24, 64, 64],  # 4b
    '4c': [512, 128, 128, 256, 24, 64, 64],  # 4c
    '4d': [512, 112, 144, 288, 32, 64, 64],  # 4d
    '4e': [528, 256, 160, 320, 32, 128, 128],  # 4e
    '5a': [832, 256, 160, 320, 32, 128, 128],  # 5a
    '5b': [832, 384, 192, 384, 48, 128, 128]  # 5b
}


class InceptionBlockSelf(nn.Module):
    def __init__(self, channel: list):
        super(InceptionBlockSelf, self).__init__()
        in_channels = channel[0]
        filter_num_1x1 = channel[1]
        filter_num_3x3reduce = channel[2]
        filter_num_3x3 = channel[3]
        filter_num_5x5reduce = channel[4]
        filter_num_5x5 = channel[5]
        filter_num_pool_proj = channel[6]
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, filter_num_1x1, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(filter_num_1x1),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, filter_num_3x3reduce, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(filter_num_3x3reduce),
            nn.ReLU(),
            nn.Conv2d(filter_num_3x3reduce, filter_num_3x3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(filter_num_3x3),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, filter_num_5x5reduce, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(filter_num_5x5reduce),
            nn.ReLU(),
            nn.Conv2d(filter_num_5x5reduce, filter_num_5x5, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(filter_num_5x5),
            nn.ReLU()
        )
        self.mp_conv1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, filter_num_pool_proj, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(filter_num_pool_proj),
            nn.ReLU()
        )

    def forward(self, in_put):
        out1x1 = self.conv1x1(in_put)
        out3x3 = self.conv3x3(in_put)
        out5x5 = self.conv5x5(in_put)
        outmp1x1 = self.mp_conv1x1(in_put)

        output = torch.cat([out1x1, out3x3, out5x5, outmp1x1], 1)
        return output


class InceptionV1(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(InceptionV1, self).__init__()

        if num_classes == 10:  # 类别为10对应Mnist或者CIFAR10数据集
            self.begin = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1)),
                nn.BatchNorm2d(192),
                nn.ReLU(True)
            )
        else:
            self.begin = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d((3, 3), stride=2, padding=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU()
            )

        self.maxpool = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.inception3a = InceptionBlockSelf(inception_layer_config['3a'])
        self.inception3b = InceptionBlockSelf(inception_layer_config['3b'])
        self.inception4a = InceptionBlockSelf(inception_layer_config['4a'])
        self.inception4b = InceptionBlockSelf(inception_layer_config['4b'])
        self.inception4c = InceptionBlockSelf(inception_layer_config['4c'])
        self.inception4d = InceptionBlockSelf(inception_layer_config['4d'])
        self.inception4e = InceptionBlockSelf(inception_layer_config['4e'])
        self.inception5a = InceptionBlockSelf(inception_layer_config['5a'])
        self.inception5b = InceptionBlockSelf(inception_layer_config['5b'])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, in_put):
        # print(input[0][0])
        output = self.begin(in_put)
        # print(output[0][0])
        output = self.inception3a(output)
        # print(output[0][0])
        output = self.inception3b(output)
        # print(output[0][0])
        output = self.maxpool(output)
        # print(output[0][0])
        output = self.inception4a(output)
        # print(output[0][0])
        output = self.inception4b(output)
        # print(output[0][0])
        output = self.inception4c(output)
        # print(output[0][0])
        output = self.inception4d(output)
        # print(output[0][0])
        output = self.inception4e(output)
        # print(output[0][0])
        output = self.maxpool(output)
        # print(output[0][0])
        output = self.inception5a(output)
        # print(output[0][0])
        output = self.inception5b(output)
        # print(output[0][0])
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        # output = self.dropout(output)
        output = self.fc(output)
        # print(output)
        return output


if __name__ == '__main__':
    model = InceptionV1(num_classes=10)
    # model = InceptionNet(10, batch_norm=True)
    if torch.cuda.is_available():
        model.cuda()
    torchsummary.summary(model, input_size=(3, 32, 32))
    # net = InceptionNet(num_classes=10, batch_norm=True)
    # print(net)
