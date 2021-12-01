# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
import torch.nn.functional as F

class KLLeNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.conv1 = nn.Conv2d(params.in_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        #for cifar it's 5x5
        #for mnist it's 4x4
        #for lfw it's 9x6
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, params.num_classes)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x=self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x=self.dropout1(x)
        x = x.view(-1, 50 * 4 * 4)
        x = F.relu(self.fc1(x))
        x=self.dropout2(x)
        x = self.fc2(x)
        return x
