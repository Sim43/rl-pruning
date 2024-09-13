import torch
import torch.nn as nn
import CONSTANTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc3 = nn.Linear(CONSTANTS.RNN_output_dims, output_dim)
        self.act5 = nn.ReLU()

    def forward(self, x):
        x = self.act5(self.fc3(x))
        return x