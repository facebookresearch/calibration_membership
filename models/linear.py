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
