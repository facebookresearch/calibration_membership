# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
import torch.nn.functional as F
from .lenet import LeNet
from .KLlenet import KLLeNet
from .lstmlm import LSTMLM
from .alexnet import AlexNet
from .linear import LinearNet
from .mlp import MLP
import torchvision.models as models 

def build_model(params):
    if params.architecture == "lenet":
        return LeNet(params)
    elif params.architecture == "kllenet":
        return KLLeNet(params)
    elif params.architecture == "linear":
        return LinearNet(params)
    elif params.architecture == "mlp":
        return MLP(params)
    elif params.architecture=="alexnet":
        return AlexNet(params)
    elif params.architecture == "lstm":
        return LSTMLM(params)
    elif params.architecture == "resnet18":
        return models.resnet18(pretrained=False)
    elif params.architecture == "smallnet":
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, params.num_classes, bias=True),
        )
    elif params.architecture == "leaks":
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, params.num_classes, bias=True),
        )
