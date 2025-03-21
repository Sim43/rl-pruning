# Reproducibility
SEED = 32

# Encoder
ENCODED_DIMS = 64 # Dimension of the encoding (Output dimension of the encoder)

K = 4 # group the output feature maps into k sets (Output dimension of the decoder)
RNN_OUTPUT_DIMS = 32 # input dimension to decoder
RNN_NUM_LAYERS = 1 # Number of stacked RNN layers. Default is 1.
RNN_HIDDEN_SIZE = 128 # The number of features in the hidden state h of the RNN

# RL
EPSILON_START = 1.0  # exploration probability at start
EPSILON_END = 0.1  # minimum exploration probability
EPSILON_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
GAMMA = 0.999 # discount factor
BATCH_SIZE = 32
LEARNING_RATE_DQN = 1e-4
ALPHA = 0.1 # to rescale Lcls into a proper range, since it varies for different networks and tasks. α was set such that the average αLcls is approximately 1
PENALTY = -0.1 # negative penalty that can be manually set
NUM_EPISODES = 3 # numer of episodes in RL
TAU = 0.005 # TAU is the update rate of the target network

# CNN
NUM_CNN_LAYERS = 4
EPOCHS_CNN = 15
MOMENTUM = 0.9
LEARNING_RATE_CNN = 0.001 # Learning rate for the CNN optimizer
FEATURE_MAP_SIZE = 32
# CNN Finetune
FINETUNE_STEPS = 5