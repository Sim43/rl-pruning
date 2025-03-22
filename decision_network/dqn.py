import torch
import torch.nn as nn
from CONSTANTS import RNN_HIDDEN_SIZE, K, ENCODED_DIMS
from typing import List, Optional, Tuple

class DQN(nn.Module):
    """
    Deep Q-Network with an encoder-RNN-decoder architecture.

    This processes feature maps from different layers of a CNN, encodes them into a fixed-size embedding,
    processes the embeddings sequentially using an RNN, and produces Q-values for a set of actions.
    """
    def __init__(self, layer_channels: List[int], embedding_size: int = ENCODED_DIMS,
                 rnn_hidden_size: int = RNN_HIDDEN_SIZE, num_actions: int = K) -> None:
        
        super(DQN, self).__init__()
        # Encoder layers (one per feature map)
        self.encoders = nn.ModuleList([
            nn.Linear(nc, embedding_size) for nc in layer_channels
        ])
        # RNN to process embeddings sequentially
        self.rnn = nn.GRU(embedding_size, rnn_hidden_size, batch_first=True)
        # Decoder to produce Q-values
        self.decoder = nn.Linear(rnn_hidden_size, num_actions)

    def forward(self, feature_map: torch.Tensor, layer_idx: int, prev_rnn_hidden_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch of feature map.
        Args:
            feature_map (torch.Tensor): A batch of feature map of shape [batch, channels, height, width].
            layer_idx (int): Index of the current layer (used to select the correct encoder).
            prev_rnn_hidden_state (Optional[torch.Tensor]): Previous RNN hidden state. If None, it is initialized to zeros.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - q_values: Q-values.
                - next_hidden_state: New RNN hidden state.
        """
        # Adaptive average pooling
        pooled = nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).squeeze()  # [batch, channels]
        
        # Project to embedding size using the corresponding encoder
        embedding = self.encoders[layer_idx](pooled) # [batch, embedding_size]
        embedding = embedding.unsqueeze(1)
        
        # Process through RNN
        if prev_rnn_hidden_state is None:
            prev_rnn_hidden_state = torch.zeros(1, embedding.size(0), self.rnn.hidden_size,
                                      device=embedding.device, dtype=embedding.dtype)

        _, new_hidden = self.rnn(embedding, prev_rnn_hidden_state)
        
        # Use final hidden state to compute Q-values
        q_values = self.decoder(new_hidden.squeeze(0))  # [batch, num_actions]

        return q_values, new_hidden