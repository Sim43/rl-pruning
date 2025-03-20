import numpy as np
import random
from collections import namedtuple, deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision.models.feature_extraction import create_feature_extractor
from torcheval.metrics.functional import multiclass_accuracy

from decision_network.encoder_change_dims import ChangeDims
from decision_network.DQN import DQN

from utils import get_device

from CONSTANTS import EPSILON_END, EPSILON_START, NUM_EPISODES, BATCH_SIZE, K, ALPHA, PENALTY, NUM_CNN_LAYERS, GAMMA, LEARNING_RATE_DQN, TAU

# if GPU is to be used
device = get_device()
# torch.autograd.set_detect_anomaly(True)

steps_done = 0
# e-greedy linear annealing
# CHECK WHETHER BATCH ITERATE OR ALL TOGETHER
def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_END - (EPSILON_START - EPSILON_END)*steps_done/NUM_EPISODES
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            x = policy_net(state).max(1).indices
            x[x==0] = 1

            # x = torch.add(x, 1)
            return x
    else:
        # NOT SURE ABOUT LOW AND HIGH. NEED TO CHECK
        return torch.randint(low = 1, high = K, size=(BATCH_SIZE,), device = device)

def reward_func(action, cur_CNN_layer, alpha = ALPHA, p = PENALTY, L_cls = None):
    if cur_CNN_layer == NUM_CNN_LAYERS - 2:
        reward = -alpha*L_cls + (action)*p
    else:
        reward = (action) * p
    return reward

def optimize_model(state_batch, next_state_batch, reward_batch, action_batch, policy_net, target_net, optimizer):
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(1,-1))
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    change_dim = ChangeDims(next_state_batch.shape[1], out_dims=state_batch.shape[1]).to(device)
    next_state_batch = change_dim(next_state_batch)
    next_state_values = target_net(next_state_batch).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    state_action_values = state_action_values.squeeze()
    
    expected_state_action_values=expected_state_action_values.squeeze()

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def prune_and_get_next_state(cnn_model, action, inputs, layer_name, output_layer_name):
    intermed_model = create_feature_extractor(cnn_model, return_nodes=[output_layer_name])
    inputs = inputs.to(device)

    next_state_batch = []

    for ind, a in enumerate(action):
        # Prune specific channels in the desired layers
        for name, module in cnn_model.named_modules():
            if name == layer_name:
                prune.identity(module, name='weight')

        # To get the layer to prune
        module_to_prune = 0
        for name, module in cnn_model.named_modules():
            if name == f'{layer_name}':
                module_to_prune = module

        # Perform pruning
        for name, module in cnn_model.named_buffers():
            if name == f'{layer_name}.weight_mask':
                num_of_channels = module.shape[0]

                group_size = num_of_channels/K
                x = torch.ones(round(group_size*a.item()), *module.shape[1:])
                y = torch.zeros(num_of_channels - round(group_size*a.item()), *module.shape[1:])
                x = torch.cat((x,y), dim=0).to(device)
                
                module_to_prune.register_buffer(name='weight_mask', tensor=x, persistent=True)

        # Generate next state
        next_state = intermed_model(inputs[ind])[output_layer_name]

        next_state = nn.functional.avg_pool2d(next_state, next_state.size()[2:]).squeeze()
        next_state_batch.append(next_state.cpu().detach())
    return torch.stack(next_state_batch, dim=0).to(device)

def driver(trainloader, cnn_model, cur_CNN_layer, layer_names, conv_layers, L_cls = None):
    intermed_model = create_feature_extractor(cnn_model, return_nodes=[layer_names[cur_CNN_layer]])

    # Forward
    inputs, classes = next(iter(trainloader))
    inputs = inputs.to(device)
    intermediate_outputs = intermed_model(inputs)
    # print(layer_names[cur_CNN_layer])
    intermediate_outputs = intermediate_outputs[layer_names[cur_CNN_layer]]
    intermediate_outputs = nn.functional.avg_pool2d(intermediate_outputs, intermediate_outputs.size()[2:]).squeeze()

    # print(intermediate_outputs.shape)

    # Get the number of state observations
    n_observations = intermediate_outputs.shape[1]

    # print("n_observations: ",n_observations)

    policy_net = DQN(n_observations).to(device)
    target_net = DQN(n_observations).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE_DQN, amsgrad=True)

    mean_L = []
    mean_list=[]
    mean_acc = []
    for j in tqdm(range(NUM_EPISODES)):
        # Initialize the environment and get its state
        inputs, classes = next(iter(trainloader))

        states = intermed_model(inputs.to(device))
        state = states[layer_names[cur_CNN_layer]]
        # Global pool
        state = nn.functional.avg_pool2d(state, state.size()[2:]).squeeze()
        # print("DIONSDIN")
        # print(state)

        # sample e-greedy actions
        action = select_action(state, policy_net)
        # print("Action:")
        # print(action)
        mean = torch.mean(action, dtype=torch.float).item()
        mean_list.append(mean)
        next_states = prune_and_get_next_state(cnn_model, action, inputs, conv_layers[cur_CNN_layer+1], layer_names[cur_CNN_layer + 1])

        # Calculate L_cls if last layer
        if cur_CNN_layer == NUM_CNN_LAYERS - 2:
            loss_fn = nn.CrossEntropyLoss()
            y_pred = cnn_model(inputs.to(device))
            loss_cnn = loss_fn(y_pred, classes.to(device))
            L_cls = loss_cnn.item()
            model_acc = multiclass_accuracy(y_pred, classes.to(device))
            mean_L.append(L_cls)
            mean_acc.append(model_acc)

        # Compute corresponding rewards
        reward = reward_func(action, cur_CNN_layer, L_cls=L_cls)
        
        # Perform one step of the optimization (on the policy network)
        # state_batch, reward_batch, action_batch, policy_net, target_net, optimizer
        optimize_model(state_batch=state, next_state_batch=next_states, reward_batch=reward, action_batch=action, policy_net=policy_net, target_net=target_net, optimizer=optimizer)
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

        target_net.load_state_dict(target_net_state_dict)

    output_img_size = 32
    group_size = getattr(cnn_model, f"conv{{}}".format(layer_names[cur_CNN_layer+1][-1])).in_channels/K
    in_channels_mean = group_size * sum(mean_list)//len(mean_list)
    
    multiplications = getattr(cnn_model, f"conv{{}}".format(layer_names[cur_CNN_layer+1][-1])).kernel_size[0]\
                      *getattr(cnn_model, f"conv{{}}".format(layer_names[cur_CNN_layer+1][-1])).kernel_size[1]\
                      *in_channels_mean\
                      *getattr(cnn_model, f"conv{{}}".format(layer_names[cur_CNN_layer+1][-1])).out_channels\
                      *(output_img_size**2)
    
    print('Complete')
    return multiplications