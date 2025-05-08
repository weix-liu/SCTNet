# -*- coding: utf-8 -*-
# @Time    : 2024/7/21 下午5:11
# @Author  : xiaoshun
# @Email   : 3038523973@qq.com
# @File    : scnn.py
# @Software: PyCharm

# 论文地址:https://www.sciencedirect.com/science/article/abs/pii/S0924271624000352?via%3Dihub#fn1

import torch
import torch.nn as nn
import torch.nn.functional as F

class SCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.conv3 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x




from thop import profile
from thop import clever_format
if __name__ == '__main__':
    # fake_image = torch.rand(1, 3, 256, 256)
    model = SCNN(num_classes=2)
    # output = model(fake_image)
    # print(output.size())

    input = torch.randn(1, 3, 256, 256)
    macs, params = profile(model, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs: {macs}")
    print(f"Parameters: {params}")

