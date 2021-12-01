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
