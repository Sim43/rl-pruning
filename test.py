import torch
import matplotlib.pyplot as plt
from utils import load_dataset, get_device
from backbone_cnn.cnn_train import get_trained_CNN_model
from decision_network.pruner_manager import PrunerManager
from decision_network.dqn_trainer import DQNTrainer
from decision_network.cnn_finetuner import CNNFineTuner
import os

def plot_images(images, titles, ncols=2):
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

def main():
    device = get_device()
    # Load dataset
    trainloader, testloader = load_dataset(dataset_name="FashionMNIST")
    data_iter = iter(testloader)
    images, labels = next(data_iter)
    images, labels = images[:2], labels[:2]
    images, labels = images.to(device), labels.to(device)

    # Load original CNN model
    cnn_path = os.path.join("models", "FashionMNIST_cnn_model.pth")
    dqn_path = os.path.join("models", "dqn_model.pth")
    model, _ = get_trained_CNN_model("FashionMNIST", trainloader, testloader, device)
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

if __name__ == "__main__":
    main()