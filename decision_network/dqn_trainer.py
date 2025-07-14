import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import *
from decision_network.dqn import DQN
from decision_network.pruner_manager import PrunerManager

class DQNTrainer:
    """
    A class to train the Deep Q-Network (DQN) for runtime neural pruning.
    """
    def __init__(self, pruner_manager: PrunerManager, device: torch.device) -> None:
        """
        Initialize the DQNTrainer.

        Args:
            pruner_manager (PrunerManager): Manages pruning operations.
            device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
        """
        self.pruner_manager = pruner_manager
        self.target_net = DQN([64,128,64,32]).to(self.pruner_manager.device)
        self.optimizer = optim.AdamW(self.pruner_manager.policy_net.parameters(), lr=LEARNING_RATE_DQN, amsgrad=True)
        self.steps_done = 0
        self.prev_rnn_hidden_state = None
        self.device = device

    def save_model(self, path: str = "models/dqn_model.pth"):
        torch.save(self.pruner_manager.policy_net.state_dict(), path)
        print(f"✅ Saved DQN model to {path}")

    def predict_action(self, state: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actions for pruning using the policy network.

        Args:
            state (torch.Tensor): The current state (feature map of previous CNN layer).
            layer_idx (int): The index of the current CNN layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - actions: The predicted actions for pruning the current CNN layer.
                - cur_hidden_state: The updated hidden state of the RNN.
        """
        self.pruner_manager.policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            q_values, cur_hidden_state = self.pruner_manager.policy_net(state, layer_idx, self.prev_rnn_hidden_state)
            actions = q_values.max(1).indices
            return actions, cur_hidden_state
    
    def _e_greedy_actions(self, state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Select actions using the epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state (feature map of previous CNN layer).
            layer_idx (int): The index of the current CNN layer.

        Returns:
            actions (torch.Tensor): The selected actions.
        """
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / (NUM_EPISODES/3))
        
        # Generate random actions for the entire batch
        random_actions = torch.randint(low=0, high=K-1, size=(BATCH_SIZE,), device=self.device)
        
        # Predict actions using the model (exploitation)
        predicted_actions = self.predict_action(state, layer_idx)[0]  # Assuming this returns the action tensor
        
        # Decide whether to explore or exploit for each action in the batch
        explore_mask = torch.rand(BATCH_SIZE, device=self.device) < eps_threshold
        return torch.where(explore_mask, random_actions, predicted_actions)
    
    def _reward_func(self, action: torch.Tensor, cur_layer_idx: int, L_cls: Optional[float] = None,
                     alpha: float = ALPHA, p: float = PENALTY) -> torch.Tensor:
        """
        Compute the reward for the given action and layer.

        Args:
            action (torch.Tensor): The actions taken for pruning.
            cur_layer_idx (int): The index of the current CNN layer.
            L_cls (Optional[float]): The classification loss (used for the last layer).
            alpha (float): Weight for the classification loss. Defaults to ALPHA.
            p (float): Penalty factor for pruning. Defaults to PENALTY.

        Returns:
            reward (torch.Tensor): The computed reward.
        """
        if cur_layer_idx == NUM_CNN_LAYERS - 1:
            return -alpha*L_cls + action*p
        return action * p

    def _optimize_model(self, state_batch: torch.Tensor, next_state_batch: torch.Tensor,
                        reward_batch: torch.Tensor, action_batch: torch.Tensor, cur_layer_idx: int) -> float:
        """
        Perform one optimization step for the DQN.

        Args:
            state_batch (torch.Tensor): The current states (feature maps).
            next_state_batch (torch.Tensor): The next states after pruning.
            reward_batch (torch.Tensor): The rewards for the current actions.
            action_batch (torch.Tensor): The actions taken for pruning.
            cur_layer_idx (int): The index of the current CNN layer.

        Returns:
            loss (float): The loss value for the optimization step.
        """
        self.pruner_manager.policy_net.train()
        state_action_values, cur_hidden_state = self.pruner_manager.policy_net(
                                                    state_batch,
                                                    cur_layer_idx - 1,
                                                    self.prev_rnn_hidden_state
                                                )
        state_action_values = state_action_values.gather(1, action_batch.view(1,-1)).squeeze()

        # Compute V(s_{t+1}) for all next states
        self.target_net.eval()
        with torch.no_grad():
            next_state_values, _ = self.target_net(next_state_batch, cur_layer_idx, cur_hidden_state)
            next_state_values = next_state_values.max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.squeeze()

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pruner_manager.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item()
    
    def _soft_update_target_network(self) -> None:
        """
        Soft update the target network weights using the policy network weights.
        Formula: θ′ ← τ θ + (1 - τ)θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.pruner_manager.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def train_step(self, inputs: torch.Tensor, classes: torch.Tensor, cur_layer_ind: int) -> None:
        """
        Perform one training step for the DQN.

        Args:
            inputs (torch.Tensor): The input batch of images.
            classes (torch.Tensor): The target labels for the input batch.
            cur_layer_ind (int): The index of the current CNN layer.
        """
        self.steps_done += 1

        # Pass input till the current CNN layer. This is the state for the DQN.
        state = self.pruner_manager.continue_forward_pass(intermediate_output=inputs, start_layer="conv1", end_layer=self.pruner_manager.return_nodes[cur_layer_ind - 1])

        # sample e-greedy actions
        action = self._e_greedy_actions(state, cur_layer_ind - 1)
        
        next_states = self.pruner_manager.prune_and_get_next_state(action, state, cur_layer_ind)

        L_cls = None

        # Calculate L_cls if last layer
        if cur_layer_ind == NUM_CNN_LAYERS - 1:
            loss_fn = nn.CrossEntropyLoss()
            y_pred = self.pruner_manager.continue_forward_pass(intermediate_output=next_states, start_layer="pool")
            loss_cnn = loss_fn(y_pred, classes)
            L_cls = loss_cnn.item()

        # Compute corresponding rewards
        reward = self._reward_func(action, cur_layer_ind, L_cls=L_cls)
        
        # Perform one step of the optimization (on the policy network)
        self._optimize_model(
            state_batch=state,
            next_state_batch=next_states,
            reward_batch=reward,
            action_batch=action,
            cur_layer_idx=cur_layer_ind
            )
        
        # Soft update the target network. This stabilizes the DQN training.
        self._soft_update_target_network()