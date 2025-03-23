from typing import List
import torch
from torch import nn
from decision_network.dqn import DQN
from CONSTANTS import NUM_CNN_LAYERS, K
from utils import get_multiplications_per_conv_layer
import torch.nn.utils.prune as prune

class PrunerManager:
    """
    A class to manage pruning operations for the CNN model using the trained policy network.
    """
    def __init__(self, cnn_model: nn.Module, img_dim: int, conv_layers: List[str], return_nodes: List[str], device: torch.device):
        """
        Initialize the PrunerManager.

        Args:
            cnn_model (nn.Module): The CNN model to be pruned.
            img_dim (int): The dimension of images in the dataset.
            conv_layers (List[str]): Names of the convolutional layers in the CNN model.
            return_nodes (List[str]): Names of the return nodes for intermediate outputs.
            device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
        """
        self.cnn_model = cnn_model
        self.conv_layers = conv_layers
        self.return_nodes = return_nodes
        self.img_dim = img_dim
        self.device = device
        self.policy_net = DQN([64,128,64,32]).to(self.device)

    def count_multiplications_in_pruned_model(self, mean_action: List[float]) -> List[int]:
        """
        Calculate the number of multiplications for each layer in the pruned model.

        Args:
            mean_action (List[float]): A list of mean actions per batch taken for each layer during pruning.

        Returns:
            List[int]: A list of number of multiplications for each layer in the pruned model.
        """
        multiplications = []
        for i in range(NUM_CNN_LAYERS):
            conv_layer = getattr(self.cnn_model, self.conv_layers[i])

            group_size = conv_layer.in_channels / K
            in_channels_mean = int(group_size * (mean_action[i] + 1))
            
            num_of_multiplications = get_multiplications_per_conv_layer(
                                conv_layer = conv_layer,
                                output_img_size = self.img_dim,
                                in_channels = in_channels_mean
                            )
            multiplications.append(num_of_multiplications)
        return multiplications

    def prune_and_get_next_state(self, action: torch.Tensor, inputs: torch.Tensor, cur_layer_idx: int) -> torch.Tensor:
        """
        Prune the specified layer and generate the next state for the pruned model.

        Args:
            action (torch.Tensor): The actions taken for pruning.
            inputs (torch.Tensor): The intermediate input for the current layer.
            cur_layer_idx (int): The index of the current CNN layer.

        Returns:
            torch.Tensor: The next state after pruning.
        """
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
            a = a.item() + 1
            for name, module in self.cnn_model.named_buffers():
                if name == f'{self.conv_layers[cur_layer_idx]}.weight_mask':
                    num_of_channels = module.shape[0]

                    group_size = num_of_channels/K
                    x = torch.ones(round(group_size*a), *module.shape[1:])
                    y = torch.zeros(num_of_channels - round(group_size*a), *module.shape[1:])
                    x = torch.cat((x,y), dim=0).to(self.device)
                    
                    module_to_prune.register_buffer(name='weight_mask', tensor=x, persistent=True)
                    
            # Generate next state
            next_state = self.continue_forward_pass(inputs[ind], self.conv_layers[cur_layer_idx], self.return_nodes[cur_layer_idx])
            next_state_batch.append(next_state.cpu().detach())

        return torch.stack(next_state_batch, dim=0).to(self.device)
    
    def continue_forward_pass(self, intermediate_output: torch.Tensor, start_layer: str,
                              end_layer: str = "fc4", eval: bool = True) -> torch.Tensor:
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