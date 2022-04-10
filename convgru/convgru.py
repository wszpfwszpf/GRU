# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/4/8 19:06
import torch
import torch.nn as nn


# import torch.nn.functional as F


def convGru_forward(x, h_t_1):
    """GRU卷积流程
    args:
        x: input
        h_t_1: 上一层的隐含层输出值
    shape：
        x: [1, channels, width, lenth]
    """
    conv_x_z = nn.Conv2d(
        in_channels=1, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    conv_h_z = nn.Conv2d(
        in_channels=4, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    z_t = torch.sigmoid(conv_x_z(x) + conv_h_z(h_t_1))

    conv_z = nn.Conv2d(in_channels=x.size()[1] + h_t_1.size()[1], out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    z_t_v2 = torch.sigmoid(conv_z(torch.cat([x, h_t_1], 1)))


    conv_x_r = nn.Conv2d(
        in_channels=1, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    conv_h_r = nn.Conv2d(
        in_channels=4, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    r_t = torch.sigmoid((conv_x_r(x) + conv_h_r(h_t_1)))
    r_tv2  = torch.sigmoid(conv_z(torch.cat([x, h_t_1], 1)))
    conv = nn.Conv2d(
        in_channels=1, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
    conv_u = nn.Conv2d(
        in_channels=4, out_channels=4, kernel_size=(1, 1), stride=(1, 1))

    h_hat_t = torch.tanh(conv(x) + conv_u(torch.mul(r_t, h_t_1)))
    h_hat_tv2 =  torch.tanh(conv_z(torch.cat([x, torch.mul(r_t, h_t_1)], 1)))

    h_t = torch.mul((1 - z_t), h_t_1) + torch.mul(z_t, h_hat_t)
    # print(z_t)
    # print(1-z_t)
    conv_out = nn.Conv2d(
        in_channels=4, out_channels=1, kernel_size=(1, 1), stride=(1, 1))  # (hidden_size, out_size)
    y = conv_out(h_t)
    return y, h_t


x = torch.randn(1, 1, 16, 16)
h_t_1 = torch.randn(1, 4, 16, 16)
y_3, h_3 = convGru_forward(x, h_t_1)
print(y_3.size())
