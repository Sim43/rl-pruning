import pandas as pd
from CONSTANTS import NUM_CNN_LAYERS, SEED

from utils import get_multiplications_per_conv_layer, load_dataset, get_device, set_random_seed
from cnn_train import get_trained_CNN_model
from decision_network.runtime_neural_pruning import ReinforcementLearning
from tabulate import tabulate

class RuntimeNeuralPruning:
    """
    class to perform Runtime Neural Pruning and model fine-tuning in an interleaving fashion.
    """
    def __init__(self):
        # Set random seed for reproducibility
        set_random_seed(SEED)

        # Get device (GPU or CPU)
        self.device = get_device()

        # Load dataset
        self.trainloader, self.testloader = load_dataset()

        # Get the trained CNN model
        self.model, self.acc_original_model = get_trained_CNN_model(self.trainloader, self.testloader, self.device)

        # Define return nodes and convolutional layers
        self.return_nodes = ['drop1', 'drop2', 'drop3', 'drop4']
        self.conv_layers = ['conv1', 'conv2', 'conv3', 'conv4']

        # Initialize multiplications per layer and total multiplications
        self.multiplications_per_layer = [[0, 0] for _ in range(NUM_CNN_LAYERS)]
        self.multiplications_original_model = 0
        self.multiplications_pruned_model = 0
        self.decision_network_trainer = ReinforcementLearning(self.model, self.trainloader, self.testloader, self.return_nodes, self.conv_layers, self.device)

    def calculate_original_model_multiplications(self):
        for ind, layer_name in enumerate(self.conv_layers):
            conv_layer = getattr(self.model, layer_name)

            self.multiplications_per_layer[ind][0] = get_multiplications_per_conv_layer(conv_layer)
            self.multiplications_original_model += self.multiplications_per_layer[ind][0]
        # print("Total multiplications for full model for one batch:", self.multiplications_original_model)

    def prune_and_finetune_model(self):
        """
        Perform Runtime Neural Pruning and model finetuning in an interleaving fashion.
        """
        for i in range(1, NUM_CNN_LAYERS):
            self.decision_network_trainer.train_q_network(i)

            # If it's the last Conv layer, we need to fine-tune the model for a longer period
            # to ensure the model converges properly after pruning
            self.decision_network_trainer.finetune_cnn()
    
    def test_models(self):
        self.acc_pruned_model, pruned_model_multiplications = self.decision_network_trainer.test_cnn_model_with_runtime_pruning()
        for ind, multis in enumerate(pruned_model_multiplications):
            self.multiplications_pruned_model += multis
            self.multiplications_per_layer[ind][1] = multis

    def display_results(self):
        """
        Display the comparison table of multiplications between the original and pruned model.
        """
        data = {layer_name: self.multiplications_per_layer[i] for i, layer_name in enumerate(self.conv_layers)}
        data["Total"] = [self.multiplications_original_model, self.multiplications_pruned_model]

        df = pd.DataFrame(data, index=["Original Model", "Pruned Model"])

        print("Comparision of Multiplications for one batch:")
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=",.0f"))

        df_acc = pd.DataFrame({"Test Accuracy": [self.acc_original_model, self.acc_pruned_model]}, index=["Original Model", "Pruned Model"])
        print(tabulate(df_acc, headers='keys', tablefmt='fancy_grid'))

        pruned_percentage = round(self.multiplications_pruned_model/self.multiplications_original_model*100, 2)

        print(f"\n\nPruned model contains {pruned_percentage}% of the original model's multiplications.")
        print(f"\n\nThe test accuracy changed by only {abs(self.acc_pruned_model - self.acc_original_model)}")        

    def run(self):
        """
        Run the pruning pipeline.
        """
        self.calculate_original_model_multiplications()
        self.prune_and_finetune_model()
        self.test_models()
        self.display_results()
    
if __name__ == "__main__":
    pruner = RuntimeNeuralPruning()
    pruner.run()