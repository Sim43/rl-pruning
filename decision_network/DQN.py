import torch.nn as nn
import CONSTANTS
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations):
        super(DQN, self).__init__()
        self.layer1 = Encoder(in_dims=n_observations)
        self.layer2 = RNN(hidden_dim=CONSTANTS.RNN_hidden_size)
        self.layer3 = Decoder(output_dim=CONSTANTS.K)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # input 32x32, output 32x64
        x = self.layer1(x)
        x = x.unsqueeze(2)

        # input 64x64x1, output 64x32
        x = self.layer2(x)

        # input 32x32, output 32xK       
        return self.layer3(x)

class Encoder(nn.Module):
    def __init__(self, in_dims, out_dims = CONSTANTS.encoded_dims):
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims)
        self.act = nn.ReLU()

    def forward(self, x):
        # input 2048, output 128
        x = self.act(self.fc(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc = nn.Linear(CONSTANTS.RNN_output_dims, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc(x))
        return x
    
class RNN(nn.Module):
    """
    This RNN is inspired from https://www.kaggle.com/code/kanncaa1/recurrent-neural-network-with-pytorch
    
    """
    def __init__(self, hidden_dim):
        super(RNN, self).__init__()
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