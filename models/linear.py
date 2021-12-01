# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
import torch.nn.functional as F

class LinearNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.fc = nn.Linear(params.data_num_dimensions, params.num_classes)

    def forward(self, x):
        x = self.fc(x)
        x=F.softmax(x)
        return x
