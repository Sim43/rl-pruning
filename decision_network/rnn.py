# https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch

import torch
import torch.nn as nn
import CONSTANTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create RNN Model
class RNN(nn.Module):
    def __init__(self, hidden_dim):
        super(RNN, self).__init__()
        # RNN
        self.rnn = nn.RNN(1, CONSTANTS.RNN_hidden_size, CONSTANTS.RNN_num_layers, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, CONSTANTS.RNN_output_dims)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.autograd.Variable(torch.zeros(CONSTANTS.RNN_num_layers, x.size(0), CONSTANTS.RNN_hidden_size)).to(device)
        
        # One time step
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out