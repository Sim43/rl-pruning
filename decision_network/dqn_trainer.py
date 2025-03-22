import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import *
from decision_network.dqn import DQN

class DQNTrainer:
    def __init__(self, pruner_manager):
        self.pruner_manager = pruner_manager
        self.target_net = DQN([64,128,64,32]).to(self.pruner_manager.device)
        self.optimizer = optim.AdamW(self.pruner_manager.policy_net.parameters(), lr=LEARNING_RATE_DQN, amsgrad=True)
        self.steps_done = 0
        self.prev_rnn_hidden_state = None

    def predict_action(self, state, layer_idx):
        self.pruner_manager.policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values, cur_hidden_state = self.pruner_manager.policy_net(state, layer_idx, self.prev_rnn_hidden_state)
            actions = q_values.max(1).indices
            return actions, cur_hidden_state
    
    def _e_greedy_actions(self, state, layer_idx):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / (NUM_EPISODES/3))
        
        # Generate random actions for the entire batch
        random_actions = torch.randint(low=0, high=K-1, size=(BATCH_SIZE,), device=self.pruner_manager.device)
        
        # Predict actions using the model (exploitation)
        predicted_actions = self.predict_action(state, layer_idx)[0]  # Assuming this returns the action tensor
        
        # Decide whether to explore or exploit for each action in the batch
        explore_mask = torch.rand(BATCH_SIZE, device=self.pruner_manager.device) < eps_threshold
        return torch.where(explore_mask, random_actions, predicted_actions)
    
    def _reward_func(self, action, cur_layer_idx, L_cls = None, alpha = ALPHA, p = PENALTY):
        if cur_layer_idx == NUM_CNN_LAYERS - 1:
            return -alpha*L_cls + action*p
        return action * p

    def _optimize_model(self, state_batch, next_state_batch, reward_batch, action_batch, cur_layer_idx):
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
    
    def _soft_update_target_network(self):
        """
        Soft update the target network weights using the policy network weights.
        Formula: θ′ ← τ θ + (1 - τ)θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.pruner_manager.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def train_step(self, inputs, classes, cur_layer_ind):
        self.steps_done += 1
        # Initialize the environment and get its state
        
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
        # state_batch, reward_batch, action_batch, policy_net, target_net, optimizer
        self._optimize_model(
            state_batch=state,
            next_state_batch=next_states,
            reward_batch=reward,
            action_batch=action,
            cur_layer_idx=cur_layer_ind
            )
        
        self._soft_update_target_network()