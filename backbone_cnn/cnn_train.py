from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import EPOCHS_CNN, MOMENTUM, LEARNING_RATE_CNN
from backbone_cnn.cnn import CNNModel
from torch.utils.data import DataLoader

def train_cnn(model: nn.Module, trainloader: DataLoader, testloader: DataLoader, device: torch.device,
              epochs: int = EPOCHS_CNN, lr: float = LEARNING_RATE_CNN, momentum: float = MOMENTUM):
    """
    Trains the CNN model

    Args:
        model (torch.nn.Module): The model to be trained.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to perform computations on.
        epochs (int): Number of epochs to train the model. Defaults to EPOCHS_CNN.
        lr (float): Learning rate for the optimizer. Defaults to LEARNING_RATE_CNN.
        momentum (float): Momentum factor for the SGD optimizer. Defaults to MOMENTUM.

    Returns:
        accuracy (float): Returns the test accuracy of the trained model.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        # Train the model
        model.train()

        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward, backward, and then weight update
            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)

            # Zero the gradients, otherwise the gradient would be a combination of the old gradient and the new one
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        accuracy, count = 0, 0

        # Disable gradient calculation for evaluation.
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                y_pred = model(inputs)
                accuracy += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
        accuracy = round(accuracy.item()/count*100, 2)
        print(f"Epoch {epoch + 1}:\nModel Accuracy on Test Set - {accuracy}%")
        print("="*50)
    return accuracy

def get_trained_CNN_model(dataset: str, trainloader: DataLoader, testloader: DataLoader, device: torch.device,
                          weights_file_path: str = './models/', only_load: bool = False):
    """
    Load the trained CNN model if weights file exists, else train the model on the dataset.
    
    Args:
        dataset (str): Name of current dataset.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to train the model on.
        weights_file_path (str): Path to the trained model's weights. Default is './models/'.
        only_load (bool): If True, only load the model without training or evaluation.

    Returns:
        model (torch.nn.Module): The loaded (or trained) CNN model.
        accuracy (float or None): Test accuracy of the model if evaluated, otherwise None.
    """
    weights_file_path += f"{dataset}_cnn_model.pth"

    # Initialize the CNN model using dataset-specific properties
    batch = next(iter(trainloader))[0]
    in_channels, img_size = batch.shape[1], batch.shape[2]
    model = CNNModel(in_channels=in_channels, img_size=img_size).to(device)

    accuracy = None

    if os.path.isfile(weights_file_path):
        print(f"Loading pre-trained model from {weights_file_path}")
        model.load_state_dict(torch.load(weights_file_path, map_location=device))

        if only_load:
            print("Model loaded without testing.")
            return model, None

        # Evaluate the model on the test set
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (torch.argmax(outputs, 1) == labels).sum().item()
                total += labels.size(0)
        accuracy = round(correct / total * 100, 2)
        print(f"Model Accuracy on Test Set - {accuracy}%")
        print("=" * 50)
    else:
        if only_load:
            raise FileNotFoundError(f"Model weights not found at {weights_file_path}")
        print(f"Training a new model and saving it to {weights_file_path}")
        accuracy = train_cnn(model, trainloader, testloader, device)
        torch.save(model.state_dict(), weights_file_path)

    return model, accuracy