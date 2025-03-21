import torch
import torch.nn as nn
from CONSTANTS import RNN_HIDDEN_SIZE, K, ENCODED_DIMS

class DQN(nn.Module):
    def __init__(self, layer_channels, embedding_size = ENCODED_DIMS, rnn_hidden_size= RNN_HIDDEN_SIZE, num_actions = K):
        super(DQN, self).__init__()
        # Encoder layers (one per feature map)
        self.encoders = nn.ModuleList([
            nn.Linear(nc, embedding_size) for nc in layer_channels
        ])
        # RNN to process embeddings sequentially
        self.rnn = nn.GRU(embedding_size, rnn_hidden_size, batch_first=True)
        # Decoder to produce Q-values
        self.decoder = nn.Linear(rnn_hidden_size, num_actions)

    def forward(self, feature_map, layer_idx, prev_rnn_hidden_state):
        """
        Forward pass for a single feature map.
        Args:
            feature_map: A single feature map of shape [batch, channels, height, width].
            layer_idx: Index of the current layer (used to select the correct encoder).
        """
        # Global average pooling
        pooled = nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()  # [batch, channels]
        
        # Project to embedding size using the corresponding encoder
        embedding = self.encoders[layer_idx](pooled)  # [batch, embedding_size]
        embedding = embedding.unsqueeze(1)  # [batch, 1, embedding_size] for RNN input
        
        # Process through RNN
        if prev_rnn_hidden_state is None:
            prev_rnn_hidden_state = torch.zeros(1, embedding.size(0), self.rnn.hidden_size,
                                      device=embedding.device, dtype=embedding.dtype)

        _, new_hidden = self.rnn(embedding, prev_rnn_hidden_state)
        
        # Use final hidden state to compute Q-values
        q_values = self.decoder(new_hidden.squeeze(0))  # [batch, num_actions]

        return q_values, new_hidden