import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import datetime
import numpy as np
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,kernel_size, stride,padding):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=kernel_size, stride=stride,padding=padding, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Baseline3D(nn.Module):
    def __init__(self):
        super(Baseline3D, self).__init__()
        self.conv1 = nn.Conv3d(3,16,(3,7,7),(1,3,3),(1,3,3),bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()
        self.block2 = BasicBlock(16, 24, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.block3 = BasicBlock(24, 24, 3, 1, 1)
        self.block4 = BasicBlock(24, 32, (3, 3, 3), (1, 2, 2), 1)
        self.block5 = BasicBlock(32, 32, 3, 1, 1)
        self.block6 = BasicBlock(32, 48, (3, 3, 3), (1, 2, 2), 1)
        self.block7 = BasicBlock(48, 48, 3, 1, 1)
        self.block8 = BasicBlock(48, 64, (3, 3, 3), (1, 2, 2), 1)
        self.block9 = BasicBlock(64, 64, 3, 1, 1)
        self.pool = nn.AdaptiveAvgPool3d((10,1,1))
        self.fc1 = nn.Linear(640,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_a = nn.Linear(256, 128)
        self.fc4_a = nn.Linear(128,10)
        self.fc3_s = nn.Linear(256, 128)
        self.fc4_s = nn.Linear(128,10)
        self.drop = nn.Dropout(0.25)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x_a = self.fc3_a(x)
        x_a = self.relu(x_a)
        x_a = self.drop(x_a)
        x_a = self.fc4_a(x_a)
        x_s = self.fc3_s(x)
        x_s = self.relu(x_s)
        x_s = self.drop(x_s)
        x_s = self.fc4_a(x_s)
        return x_a,x_s

net = Baseline3D()

input = torch.randn(16,3,10,342,608)

angle,speed = net(input)