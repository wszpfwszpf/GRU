# -*- coding: utf-8 -*-
# Author:ZPF
# date: 2022/3/27 9:50

import torch.nn.functional as F
import torch
import torch.nn as nn

# nn.CrossEntropyLoss() 和 torch.nn.functional.cross_entropy函数的计算结果一致。

if __name__ == '__main__':
    torch.manual_seed(10)
    criterion = nn.CrossEntropyLoss()
    input = torch.randn(3,5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(input.size()[0])
    # print(target)
    # result1 = criterion(input, target)
    # result2 = F.cross_entropy(input, target)
    # print(result1)
    # print(result2)