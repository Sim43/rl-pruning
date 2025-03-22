from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import *

class CNNFineTuner:
    def __init__(self, pruner_manager, testloader):
        self.pruner_manager = pruner_manager       
        self.cnn_loss_fn = nn.CrossEntropyLoss()
        self.cnn_optimizer = optim.SGD(self.pruner_manager.cnn_model.parameters(), lr=LEARNING_RATE_CNN, momentum=MOMENTUM)
        self.testloader = testloader

    def finetune_step(self, inputs, classes):
        y_pred, _ = self.cnn_forward_pass_with_runtime_pruning(inputs, eval=False)

        loss_cnn = self.cnn_loss_fn(y_pred, classes)

        self.cnn_optimizer.zero_grad()
        loss_cnn.backward()
        self.cnn_optimizer.step()
    
    def test_cnn_model_with_runtime_pruning(self):
        print("=" * 50)
        print("Testing the pruned network!")
        # Evaluate the model on the test set
        self.pruner_manager.cnn_model.eval()
        accuracy, count, count_loop = 0, 0, 0
        mean_action = [0] * NUM_CNN_LAYERS
        # Disable gradient calculation for evaluation.
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader):
                inputs = inputs.to(self.pruner_manager.device)
                labels = labels.to(self.pruner_manager.device)

                y_pred, actions = self.cnn_forward_pass_with_runtime_pruning(inputs)
                mean_action = [a + b for a, b in zip(mean_action, actions)]

                accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
                count_loop += 1

        accuracy = round(accuracy.item() / count*100, 2)
        mean_action = [x // count_loop for x in mean_action]
        multiplications = self.pruner_manager.count_multiplications_in_pruned_model(mean_action)
        return accuracy, multiplications 
    
    def predict_action(self, state, layer_idx, prev_rnn_hidden_state):
        self.pruner_manager.policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_values, cur_hidden_state = self.pruner_manager.policy_net(state, layer_idx, prev_rnn_hidden_state)
            actions = q_values.max(1).indices
            return actions, cur_hidden_state
        
    def cnn_forward_pass_with_runtime_pruning(self, inputs, eval = True):
        prev_rnn_hidden_state = None
        mean_action = [K]
        
        inputs = self.pruner_manager.continue_forward_pass(intermediate_output=inputs, start_layer=self.pruner_manager.conv_layers[0], end_layer=self.pruner_manager.return_nodes[0], eval=eval)

        for cur_layer_ind in range(1, NUM_CNN_LAYERS):            
            # sample e-greedy actions
            action, prev_rnn_hidden_state = self.predict_action(inputs, cur_layer_ind - 1, prev_rnn_hidden_state)

            mean_action.append(torch.mean(action, dtype=torch.float).item())

            inputs = self.pruner_manager.prune_and_get_next_state(action, inputs, cur_layer_ind)

        y_pred = self.pruner_manager.continue_forward_pass(intermediate_output=inputs, start_layer="pool", eval=eval)
        return y_pred, mean_action