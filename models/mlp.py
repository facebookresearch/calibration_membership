# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc = nn.Linear(params.data_num_dimensions, 2*params.data_num_dimensions)
        self.fc2 = nn.Linear(2*params.data_num_dimensions, params.num_classes)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x
