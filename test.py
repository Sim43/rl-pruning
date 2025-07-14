import torch
import matplotlib.pyplot as plt
from utils import load_dataset, get_device
from backbone_cnn.cnn_train import get_trained_CNN_model
from decision_network.pruner_manager import PrunerManager
from decision_network.dqn_trainer import DQNTrainer
from decision_network.cnn_finetuner import CNNFineTuner
from utils import get_multiplications_per_conv_layer
import os
import pandas as pd
from tabulate import tabulate
from CONSTANTS import K

tests = 5 

def plot_images(images, titles, ncols=tests):
    plt.figure(figsize=(15, 3))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, ncols, i+1)
        if img.shape[0] == 1:  # grayscale
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img.permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_results(conv_layers, multiplications_per_layer, total_mults_original, pruned_mults_per_layer, total_mults_pruned, acc_original=None, acc_pruned=None):
    """
    Display a comparison table of multiplications between the original and pruned model.
    Optionally, also display accuracy if provided.
    """
    data = {layer_name: [multiplications_per_layer[i], pruned_mults_per_layer[i]] for i, layer_name in enumerate(conv_layers)}
    data["Total"] = [total_mults_original, total_mults_pruned]
    df = pd.DataFrame(data, index=["Original Model", "Pruned Model"])

    print("Comparison of Multiplications for processing one batch:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=",.0f"))

    if acc_original is not None and acc_pruned is not None:
        df_acc = pd.DataFrame({"Test Accuracy": [acc_original, acc_pruned]}, index=["Original Model", "Pruned Model"])
        print(tabulate(df_acc, headers='keys', tablefmt='fancy_grid'))

    pruned_percentage = round(total_mults_pruned / total_mults_original * 100, 2)
    print(f"\nPruned model contains {pruned_percentage}% of the original model's multiplications.")

def main():
    device = get_device()
    # Load dataset
    trainloader, testloader = load_dataset(dataset_name="FashionMNIST")
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    images, labels = images[:tests], labels[:tests]
    images, labels = images.to(device), labels.to(device)

    # Load original CNN model
    cnn_path = os.path.join("models", "FashionMNIST_cnn_model.pth")
    dqn_path = os.path.join("models", "dqn_model.pth")
    model, _ = get_trained_CNN_model("FashionMNIST", trainloader, testloader, device, only_load=True)
    model.load_state_dict(torch.load(cnn_path, map_location=device))
    model.eval()

    # Predict with original CNN
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    plot_images(images.cpu(), [f"Orig: {p.item()}" for p in preds])

    # Setup pruner, DQN, and CNNFineTuner
    conv_layers = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    return_nodes = [f'drop{i+1}' for i in range(len(conv_layers))]
    pruner_manager = PrunerManager(
        cnn_model=model, img_dim=images.shape[2], conv_layers=conv_layers,
        return_nodes=return_nodes, device=device
    )
    dqn_trainer = DQNTrainer(pruner_manager=pruner_manager, device=device)
    dqn_trainer.load_model(dqn_path)
    cnn_finetuner = CNNFineTuner(
        pruner_manager=pruner_manager,
        testloader=testloader,
        conv_layers=conv_layers,
        return_nodes=return_nodes,
        device=device
    )

    # Predict with pruned CNN using DQN (on the same 5 images, batched)
    with torch.no_grad():
        y_pred, _ = cnn_finetuner.cnn_forward_pass_with_runtime_pruning(images)
        pruned_preds = torch.argmax(y_pred, 1).cpu().tolist()
    plot_images(images.cpu(), [f"Pruned: {p}" for p in pruned_preds])

    # Calculate multiplications for original model
    multiplications_per_layer = []
    total_mults_original = 0
    for layer_name in conv_layers:
        conv_layer = getattr(model, layer_name)
        mults = get_multiplications_per_conv_layer(conv_layer, images.shape[2])
        multiplications_per_layer.append(mults)
        total_mults_original += mults

    # Calculate multiplications for pruned model
    with torch.no_grad():
        _, mean_actions = cnn_finetuner.cnn_forward_pass_with_runtime_pruning(images)
    pruned_mults_per_layer = []
    total_mults_pruned = 0
    for i, layer_name in enumerate(conv_layers):
        conv_layer = getattr(model, layer_name)
        if i == 0:
            pruned_mults = get_multiplications_per_conv_layer(conv_layer, images.shape[2])
        else:
            pruned_mults = get_multiplications_per_conv_layer(conv_layer, images.shape[2]) * (mean_actions[i] / K)
        pruned_mults_per_layer.append(int(pruned_mults))
        total_mults_pruned += pruned_mults

    # Optionally, compute accuracy for original and pruned model on this batch
    acc_original = (preds.cpu() == labels.cpu()).float().mean().item() * 100
    acc_pruned = (torch.tensor(pruned_preds) == labels.cpu()).float().mean().item() * 100

    # Display results
    display_results(
        conv_layers,
        multiplications_per_layer,
        total_mults_original,
        pruned_mults_per_layer,
        int(total_mults_pruned),
        acc_original=acc_original,
        acc_pruned=acc_pruned
    )

if __name__ == "__main__":
    main()



# fashion_mnist_class_map = {
#     0: "T-shirt/top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle boot",
# }
