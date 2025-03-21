import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from decision_network.DQN import DQN
from utils import get_device, get_multiplications_per_conv_layer

from CONSTANTS import EPSILON_DECAY, EPSILON_END, EPSILON_START, FINETUNE_STEPS, LEARNING_RATE_CNN, MOMENTUM, NUM_EPISODES, BATCH_SIZE, K, ALPHA, PENALTY, NUM_CNN_LAYERS, GAMMA, LEARNING_RATE_DQN, TAU
# torch.autograd.set_detect_anomaly(True)

class ReinforcementLearning():
    def __init__(self, cnn_model, trainloader, testloader, return_nodes, conv_layers, device):
        self.steps_done = 0
        self.cnn_model = cnn_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_iterator = iter(trainloader)

        self.device = device
        self.return_nodes = return_nodes
        self.conv_layers = conv_layers
        self.policy_net = DQN([64,128,64,32]).to(self.device)
        self.target_net = DQN([64,128,64,32]).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE_DQN, amsgrad=True)
        self.prev_rnn_hidden_state = None

        self.cnn_loss_fn = nn.CrossEntropyLoss()
        self.cnn_optimizer = optim.SGD(self.cnn_model.parameters(), lr=LEARNING_RATE_CNN, momentum=MOMENTUM)

    def _predict_action(self, state, layer_idx):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values, cur_hidden_state = self.policy_net(state, layer_idx, self.prev_rnn_hidden_state)
            actions = q_values.max(1).indices
            actions[actions==0] = 1
            return actions, cur_hidden_state

    def _e_greedy_actions(self, state, layer_idx):
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
        
        if random.random() > eps_threshold:
            return self._predict_action(state, layer_idx)[0]
        return torch.randint(low = 1, high = K, size=(BATCH_SIZE,), device = self.device)

    def _reward_func(self, action, cur_layer_idx, L_cls = None, alpha = ALPHA, p = PENALTY):
        if cur_layer_idx == NUM_CNN_LAYERS - 1:
            return -alpha*L_cls + action*p
        return action * p

    def _optimize_model(self, state_batch, next_state_batch, reward_batch, action_batch, cur_layer_idx):
        state_action_values, cur_hidden_state = self.policy_net(
                                                    state_batch,
                                                    cur_layer_idx - 1,
                                                    self.prev_rnn_hidden_state
                                                )
        state_action_values = state_action_values.gather(1, action_batch.view(1,-1)).squeeze()

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values, _ = self.target_net(next_state_batch, cur_layer_idx, cur_hidden_state)
            next_state_values = next_state_values.max(1).values

            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            expected_state_action_values = expected_state_action_values.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item()

    def _prune_and_get_next_state(self, action, inputs, cur_layer_idx):
        next_state_batch = []

         # Prune specific channels in the desired layers
        module_to_prune = None
        for name, module in self.cnn_model.named_modules():
            if name == self.conv_layers[cur_layer_idx]:
                prune.identity(module, name='weight')
                module_to_prune = module
                break

        for ind, a in enumerate(action):
            # Perform pruning
            for name, module in self.cnn_model.named_buffers():
                if name == f'{self.conv_layers[cur_layer_idx]}.weight_mask':
                    num_of_channels = module.shape[0]

                    group_size = num_of_channels/K
                    x = torch.ones(round(group_size*a.item()), *module.shape[1:])
                    y = torch.zeros(num_of_channels - round(group_size*a.item()), *module.shape[1:])
                    x = torch.cat((x,y), dim=0).to(self.device)
                    
                    module_to_prune.register_buffer(name='weight_mask', tensor=x, persistent=True)
                    
            # Generate next state
            next_state = self._continue_forward_pass(inputs[ind], self.conv_layers[cur_layer_idx], self.return_nodes[cur_layer_idx])
            next_state_batch.append(next_state.cpu().detach())

        return torch.stack(next_state_batch, dim=0).to(self.device)
    
    def _continue_forward_pass(self, intermediate_output, start_layer, end_layer = "fc4", eval = True):
        """
        Continue the forward pass from the intermediate output to the final output.

        Args:
            intermediate_output (torch.Tensor): Output from the intermediate layer.
            start_layer (str): Name of the layer to start the forward pass from.
            end_layer (str): Name of the layer to return the output from. Default is fc4 if you want the final output.
            eval (bool): Set Model to eval mode if True
        Returns:
            torch.Tensor: output from end layer.
        """
        # Get the layers after the intermediate layer
        layers = []
        capture = False
        for name, module in self.cnn_model.named_modules():
            if name == start_layer:
                capture = True
            if capture:
                layers.append(module)
                if name == end_layer:
                    break

        if eval:
            self.cnn_model.eval()

            with torch.no_grad():
                # Pass the intermediate output through the remaining layers
                for layer in layers:
                    intermediate_output = layer(intermediate_output)
        else:
            self.cnn_model.train()
            for layer in layers:
                
                intermediate_output = layer(intermediate_output)
        return intermediate_output
    
    def _soft_update_target_network(self):
        """
        Soft update the target network weights using the policy network weights.
        Formula: θ′ ← τ θ + (1 - τ)θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def _cnn_forward_pass_with_runtime_pruning(self, inputs, eval = True):
        self.prev_rnn_hidden_state = None
        mean_action = [K]
        
        inputs = self._continue_forward_pass(inputs, self.conv_layers[0], self.return_nodes[0], eval=eval)

        for cur_layer_ind in range(1, NUM_CNN_LAYERS):            
            # sample e-greedy actions
            action, cur_hidden_state = self._predict_action(inputs, cur_layer_ind - 1)
            self.prev_rnn_hidden_state = cur_hidden_state

            mean_action.append(torch.mean(action, dtype=torch.float).item())

            inputs = self._prune_and_get_next_state(action, inputs, cur_layer_ind)

        y_pred = self._continue_forward_pass(inputs, start_layer="pool", eval=eval)
        return y_pred, mean_action
    
    def test_cnn_model_with_runtime_pruning(self):
        print("=" * 50)
        print("Testing the pruned network!")
        # Evaluate the model on the test set
        self.cnn_model.eval()
        accuracy, count, count_loop = 0, 0, 0
        mean_action = [0] * NUM_CNN_LAYERS
        # Disable gradient calculation for evaluation.
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                y_pred, actions = self._cnn_forward_pass_with_runtime_pruning(inputs)
                mean_action = [a + b for a, b in zip(mean_action, actions)]

                accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
                count_loop += 1

        accuracy = round(accuracy.item() / count*100, 2)
        mean_action = [x // count_loop for x in mean_action]
        multiplications = self._count_multiplications_in_pruned_model(mean_action)
        return accuracy, multiplications
    
    def _count_multiplications_in_pruned_model(self, mean_action):
        multiplications = []
        for i in range(NUM_CNN_LAYERS):
            conv_layer = getattr(self.cnn_model, self.conv_layers[i])

            group_size = conv_layer.in_channels / K
            in_channels_mean = int(group_size * mean_action[i])
            
            num_of_multiplications = get_multiplications_per_conv_layer(
                                conv_layer = conv_layer,
                                in_channels = in_channels_mean
                            )
            multiplications.append(num_of_multiplications)
        return multiplications

    def train_q_network(self, cur_layer_ind):
        print("="*50)
        print("Training Deep Q Network!")
        track_loss = []
        self.steps_done = 0
        for _ in tqdm(range(NUM_EPISODES)):
            self.steps_done += 1
            # Initialize the environment and get its state
            try:
                inputs, classes = next(self.train_iterator)
            except StopIteration:
                # Reinitialize the iterator when it reaches the end
                self.train_iterator = iter(self.trainloader)
                inputs, classes = next(self.train_iterator)
                
            inputs, classes = inputs.to(self.device), classes.to(self.device)

            state = self._continue_forward_pass(inputs, "conv1", self.return_nodes[cur_layer_ind - 1])

            # sample e-greedy actions
            action = self._e_greedy_actions(state, cur_layer_ind - 1)
            
            next_states = self._prune_and_get_next_state(action, state, cur_layer_ind)

            L_cls = None

            # Calculate L_cls if last layer
            if cur_layer_ind == NUM_CNN_LAYERS - 1:
                loss_fn = nn.CrossEntropyLoss()
                
                y_pred = self._continue_forward_pass(next_states, start_layer="pool")

                loss_cnn = loss_fn(y_pred, classes)

                L_cls = loss_cnn.item()

            # Compute corresponding rewards
            reward = self._reward_func(action, cur_layer_ind, L_cls=L_cls)
            
            # Perform one step of the optimization (on the policy network)
            # state_batch, reward_batch, action_batch, policy_net, target_net, optimizer
            track_loss.append(
                self._optimize_model(
                    state_batch=state,
                    next_state_batch=next_states,
                    reward_batch=reward,
                    action_batch=action,
                    cur_layer_idx=cur_layer_ind
                    )
                )
            
            self._soft_update_target_network()

        return track_loss

    def finetune_cnn(self):
        print("="*50)
        print("Finetuning the CNN With Runtime Neural Pruning!")
        for _ in tqdm(range(FINETUNE_STEPS)):
            try:
                inputs, classes = next(self.train_iterator)
            except StopIteration:
                # Reinitialize the iterator when it reaches the end
                self.train_iterator = iter(self.trainloader)
                inputs, classes = next(self.train_iterator)
            inputs, classes = inputs.to(self.device), classes.to(self.device)

            y_pred, _ = self._cnn_forward_pass_with_runtime_pruning(inputs, eval=False)

            loss_cnn = self.cnn_loss_fn(y_pred, classes)

            self.cnn_optimizer.zero_grad()
            loss_cnn.backward()
            self.cnn_optimizer.step()