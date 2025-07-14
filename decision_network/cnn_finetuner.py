from typing import List, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from decision_network.pruner_manager import PrunerManager
from torch.utils.data import DataLoader

from CONSTANTS import *

class CNNFineTuner:
    """
    A class to handle fine-tuning of a CNN model with runtime neural pruning.
    """
    def __init__(self, pruner_manager: PrunerManager, testloader: DataLoader, conv_layers: List[str], return_nodes: List[str], device: torch.device) -> None:
        """
        Initialize the CNNFineTuner.

        Args:
            pruner_manager (PrunerManager): Manages pruning operations and the CNN model.
            testloader (DataLoader): DataLoader for the test dataset.
            conv_layers (List[str]): Names of the convolutional layers in the CNN model.
            return_nodes (List[str]): Names of the return nodes for intermediate outputs.
            device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
        """
        self.pruner_manager = pruner_manager       
        self.cnn_loss_fn = nn.CrossEntropyLoss()
        self.cnn_optimizer = optim.SGD(self.pruner_manager.cnn_model.parameters(), lr=LEARNING_RATE_CNN, momentum=MOMENTUM)
        self.testloader = testloader
        self.conv_layers = conv_layers
        self.return_nodes = return_nodes
        self.device = device

    def finetune_step(self, inputs: torch.Tensor, classes: torch.Tensor) -> None:
        """
        Perform one fine-tuning step on the CNN model.

        Args:
            inputs (torch.Tensor): Input batch of images of shape [batch_size, channels, height, width].
            classes (torch.Tensor): Target labels of shape [batch_size].
        """
        y_pred, _ = self.cnn_forward_pass_with_runtime_pruning(inputs, eval=False)

        loss_cnn = self.cnn_loss_fn(y_pred, classes)

        self.cnn_optimizer.zero_grad()
        loss_cnn.backward()
        self.cnn_optimizer.step()
    
    def test_cnn_model_with_runtime_pruning(self) -> Tuple[float, List[int]]:
        """
        Test the CNN model with Runtime Pruning using the trained Deep Q-Network on the test dataset.

        Returns:
            Tuple[float, List[int]]: A tuple containing:
                - accuracy: The test accuracy of the pruned model.
                - multiplications: A list of number of multiplications for each layer in the pruned model.
        """
        print("=" * 50)
        print("Testing the pruned network!")

        accuracy, count, count_loop = 0, 0, 0
        mean_action = [0] * NUM_CNN_LAYERS

        # Evaluate the model on the test set
        self.pruner_manager.cnn_model.eval()
        
        # Disable gradient calculation for evaluation.
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                y_pred, actions = self.cnn_forward_pass_with_runtime_pruning(inputs)
                mean_action = [a + b for a, b in zip(mean_action, actions)]

                accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
                count_loop += 1

        accuracy = round(accuracy.item() / count*100, 2)
        mean_action = [x // count_loop for x in mean_action]
        multiplications = self.pruner_manager.count_multiplications_in_pruned_model(mean_action)
        return accuracy, multiplications 
    
    def predict_action(self, state: torch.Tensor, layer_idx: int, prev_rnn_hidden_state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actions for pruning using the policy network.

        Args:
            state (torch.Tensor): The current state (feature map of previous CNN layer).
            layer_idx (int): The index of the current CNN layer.
            prev_rnn_hidden_state (Optional[torch.Tensor]): The previous hidden state of the RNN.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - actions: The predicted actions for pruning the current CNN layer.
                - cur_hidden_state: The updated hidden state of the RNN.
        """
        self.pruner_manager.policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            q_values, cur_hidden_state = self.pruner_manager.policy_net(state, layer_idx, prev_rnn_hidden_state)
            actions = q_values.max(1).indices
            return actions, cur_hidden_state
        
    def cnn_forward_pass_with_runtime_pruning(self, inputs: torch.Tensor, eval: bool = True) -> Tuple[torch.Tensor, List[float]]:
        """
        Perform a forward pass through the CNN model with runtime pruning.

        Args:
            inputs (torch.Tensor): Input batch of images.
            eval (bool): If True, set the model to evaluation mode. Default is True.

        Returns:
            Tuple[torch.Tensor, List[float]]: A tuple containing:
                - y_pred: The output predictions.
                - mean_action: A list of mean actions taken for each layer during pruning. Used to keep track of number of multiplications performed.
        """
        prev_rnn_hidden_state = None
        mean_action = [K]
        
        inputs = self.pruner_manager.continue_forward_pass(intermediate_output=inputs, start_layer=self.conv_layers[0],
                                                           end_layer=self.return_nodes[0], eval=eval)

        for cur_layer_ind in range(1, NUM_CNN_LAYERS):            
            # sample e-greedy actions
            action, prev_rnn_hidden_state = self.predict_action(inputs, cur_layer_ind - 1, prev_rnn_hidden_state)

            mean_action.append(torch.mean(action, dtype=torch.float).item())

            inputs = self.pruner_manager.prune_and_get_next_state(action, inputs, cur_layer_ind)

        y_pred = self.pruner_manager.continue_forward_pass(intermediate_output=inputs, start_layer="pool", eval=eval)
        return y_pred, mean_action