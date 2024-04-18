from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim



# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1   = nn.Linear(400, 120)
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 10)
       
        
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
       
#         return out

# class LeNetBlock(nn.Module):
#     def __init__(self):
#         super(LeNetBlock, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(6, 16, 5)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         out = F.max_pool2d(out, 2)
#         return out

# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.layer1 = LeNetBlock()
#         self.fc1   = nn.Linear(400, 120)
#         self.fc2   = nn.Linear(120, 84)
#         self.fc3   = nn.Linear(84, 10)
       
#     def forward(self, x):
#         out = self.layer1(x)
#         out = out.view(out.size(0), -1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
       
#         return out

class LeNetBlock(nn.Module):
    def __init__(self):
        super(LeNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        # out = F.max_pool2d(out, 2)
        return out

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = LeNetBlock()
        self.fc1   = nn.Linear(3136, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2   = nn.Linear(128, 10)
       
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn3(self.fc1(out)))
        out = F.relu(self.fc2(out))
       
        return out

class CifarBlock(nn.Module):
    def __init__(self):
        super(CifarBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 196, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(196, 196, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(196)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        # out = F.max_pool2d(out, 2)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        # out = F.max_pool2d(out, 2)
        return out

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.layer1 = CifarBlock()
        self.fc1   = nn.Linear(3136, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2   = nn.Linear(256, 10)
       
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.bn7(self.fc1(out)))
        out = F.relu(self.fc2(out))
       
        return out