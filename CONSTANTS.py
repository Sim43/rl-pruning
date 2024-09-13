# Encoder
encoded_dims = 64 # Dimension of the encoding (Output dimension of the encoder)

K = 4 # group the output feature maps into k sets (Output dimension of the decoder)
RNN_output_dims = 32 # input dimension to decoder
RNN_num_layers = 1 # Number of stacked RNN layers. Default is 1.
RNN_hidden_size = 128 # The number of features in the hidden state h of the RNN

# RL
epsilon_start = 1.0  # exploration probability at start
epsilon_end = 0.1  # minimum exploration probability
epsilon_decay = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
gamma = 0.999 # discount factor
batch_size = 32
learning_rate = 1e-4
alpha = 0.1 # to rescale Lcls into a proper range, since it varies for different networks and tasks. α was set such that the average αLcls is approximately 1
p = -0.1 # negative penalty that can be manually set
num_episodes = 80 # numer of episodes in RL
TAU = 0.005 # TAU is the update rate of the target network

# CNN
num_CNN_layers = 4
epochs = 20

# Finetune
num_epochs_finetune = 5
