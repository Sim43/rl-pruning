import argparse
from typing import Tuple
import pandas as pd
import torch
from tqdm import tqdm
from CONSTANTS import FINETUNE_STEPS, NUM_CNN_LAYERS, NUM_EPISODES, SEED

from utils import get_multiplications_per_conv_layer, load_dataset, get_device, set_random_seed
from backbone_cnn.cnn_train import get_trained_CNN_model
from decision_network.pruner_manager import PrunerManager
from decision_network.dqn_trainer import DQNTrainer
from decision_network.cnn_finetuner import CNNFineTuner

from tabulate import tabulate

from torch import nn

class RuntimeNeuralPruning:
    """
    A class to perform Runtime Neural Pruning (RNP) and model fine-tuning in an interleaving fashion.

    This class implements a pipeline for pruning a pre-trained CNN model using reinforcement learning (RL).
    It dynamically prunes convolutional layers during runtime, fine-tunes the model, and evaluates the
    pruned model's performance. The results, including the number of multiplications and
    accuracy, are displayed for comparison between the original and pruned models.

        Example:
        >>> pruner = RuntimeNeuralPruning()
        >>> pruner.run()

        Comparison of Multiplications for processing one image:
        ╒══════════════════╤═══════════╤═══════════╤═══════════╤═══════════╤═══════════╕
        │                  │   conv1   │   conv2   │   conv3   │   conv4   │   Total   │
        ╞══════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═══════════╡
        │ Original Model   │  451,584  │ 57,802,752│ 57,802,752│ 14,450,688│130,507,776│
        ├──────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
        │ Pruned Model     │  451,584  │ 46,061,568| 34,320,384│  2,709,504│ 83,543,040│
        ╘══════════════════╧═══════════╧═══════════╧═══════════╧═══════════╧═══════════╛
        ╒══════════════════╤═══════════════╕
        │                  │ Test Accuracy │
        ╞══════════════════╪═══════════════╡
        │ Original Model   │         90.12 │
        ├──────────────────┼───────────────┤
        │ Pruned Model     │         88.26 │
        ╘══════════════════╧═══════════════╛

        Pruned model contains 64.01% of the original model's multiplications.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize the RuntimeNeuralPruning class.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        # Set random seed for reproducibility
        set_random_seed(args.seed)

        # Get device (GPU or CPU)
        self.device = get_device()
         
        # Load dataset
        self.trainloader, self.testloader = load_dataset(dataset_name=args.dataset)
        self.train_iterator = iter(self.trainloader)

        # Get dimension of images in the dataset. Used to calculate number of multiplications.
        self.img_dim = next(iter(self.trainloader))[0].shape[3]

        # Get the trained CNN model
        self.model, self.acc_original_model = get_trained_CNN_model(args.dataset, self.trainloader, self.testloader, self.device)

        # Define return nodes and convolutional layers
        self.conv_layers = [name for name, module in self.model.named_modules() if isinstance(module, nn.Conv2d)]
        self.return_nodes = [f'drop{i+1}' for i in range(len(self.conv_layers))]

        # Initialize multiplications per layer and total multiplications
        self.multiplications_per_layer = [[0, 0] for _ in range(NUM_CNN_LAYERS)]
        self.multiplications_original_model = 0
        self.multiplications_pruned_model = 0

        # Initialize the RL Trainer        
        pruner_manager = PrunerManager(cnn_model=self.model, img_dim=self.img_dim, conv_layers=self.conv_layers, return_nodes=self.return_nodes, device=self.device)
        self.cnn_finetuner = CNNFineTuner(pruner_manager=pruner_manager, testloader=self.testloader, conv_layers=self.conv_layers, return_nodes=self.return_nodes, device=self.device)
        self.deep_q_net_trainer = DQNTrainer(pruner_manager=pruner_manager, device=self.device)

    def calculate_original_model_multiplications(self) -> None:
        """
        Calculate the number of multiplications per image for each convolutional layer in the original model.
        """
        for ind, layer_name in enumerate(self.conv_layers):
            conv_layer = getattr(self.model, layer_name)

            self.multiplications_per_layer[ind][0] = get_multiplications_per_conv_layer(conv_layer=conv_layer, output_img_size=self.img_dim)
            self.multiplications_original_model += self.multiplications_per_layer[ind][0]

    def _get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the next batch of data from the training DataLoader after moving it to specified device.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - inputs: A batch of input data of shape [batch_size, channels, height, width].
                - classes: A batch of labels of shape [batch_size].
        """
        try:
            inputs, classes = next(self.train_iterator)
        except StopIteration:
            # Reinitialize the iterator when it reaches the end
            self.train_iterator = iter(self.trainloader)
            inputs, classes = next(self.train_iterator)
        
        inputs, classes = inputs.to(self.device), classes.to(self.device)
        return inputs, classes

    def prune_and_finetune_model(self) -> None:
        """
        Perform Runtime Neural Pruning and model finetuning in an interleaving fashion.
        """
        print("Alternate training starts from the second CNN layer.")
        for i in range(1, NUM_CNN_LAYERS):
            print(f"Performing alternating training of the Deep Q-Network and CNN model. Processing Layer {i+1} of {NUM_CNN_LAYERS}")
            episodes = NUM_EPISODES

            if i == NUM_CNN_LAYERS - 1:
                episodes *= 4

            print("Training Deep Q-Network")
            for _ in tqdm(range(episodes)):
                inputs, classes = self._get_next_batch()
                self.deep_q_net_trainer.train_step(inputs, classes, i)

            print("Finetuning CNN Model with RNP Pruning")
            for _ in tqdm(range(FINETUNE_STEPS)):
                inputs, classes = self._get_next_batch()
                self.cnn_finetuner.finetune_step(inputs, classes)
            
            print("="*50)
    
    def test_models(self) -> None:
        """
        Test the model with runtime neural pruning and update the multiplication counts for each layer.

        It retrieves the accuracy of the pruned model and the number of multiplications for each layer.
        Finally, it calls `_display_results` to display the comparison between the original and pruned models.
        """
        acc_pruned_model, pruned_model_multiplications = self.cnn_finetuner.test_cnn_model_with_runtime_pruning()
        for ind, multis in enumerate(pruned_model_multiplications):
            self.multiplications_pruned_model += multis
            self.multiplications_per_layer[ind][1] = multis
        self._display_results(acc_pruned_model)

    def _display_results(self, acc_pruned_model: float) -> None:
        """
        Display the comparison table of multiplications between the original and pruned model.

        Args:
            acc_pruned_model (float): The test accuracy of the pruned model.
        """
        data = {layer_name: self.multiplications_per_layer[i] for i, layer_name in enumerate(self.conv_layers)}
        data["Total"] = [self.multiplications_original_model, self.multiplications_pruned_model]

        df = pd.DataFrame(data, index=["Original Model", "Pruned Model"])

        print("Comparision of Multiplications for processing one image:")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=",.0f"))

        df_acc = pd.DataFrame({"Test Accuracy": [self.acc_original_model, acc_pruned_model]}, index=["Original Model", "Pruned Model"])
        print(tabulate(df_acc, headers='keys', tablefmt='fancy_grid'))

        pruned_percentage = round(self.multiplications_pruned_model/self.multiplications_original_model*100, 2)

        print(f"\n\nPruned model contains {pruned_percentage}% of the original model's multiplications.")

    def run(self) -> None:
        """
        Run the pruning pipeline.
        """
        self.calculate_original_model_multiplications()
        self.prune_and_finetune_model()
        self.test_models()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Runtime Neural Pruning (RNP).")

    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")

    # Dataset choices
    dataset_choices = ["FashionMNIST", "CIFAR10"]
    parser.add_argument("--dataset", type=str, default="FashionMNIST", choices=dataset_choices, help=f"Name of dataset. Must be one of: {', '.join(dataset_choices)}.")

    parser.add_argument("--num_episodes_rl", type=int, default=NUM_EPISODES, help="Number of episodes for RL training.")
    parser.add_argument("--finetune_steps_cnn", type=int, default=FINETUNE_STEPS, help="Number of fine-tuning steps.")
    return parser.parse_args()
    
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Override constants with CLI arguments
    NUM_EPISODES = args.num_episodes_rl
    FINETUNE_STEPS = args.finetune_steps_cnn

    pruner = RuntimeNeuralPruning(args)
    pruner.run()