import torch.nn as nn

class ChangeDims(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        x = self.fc(x)
        return x