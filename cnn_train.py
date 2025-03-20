from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim

from CONSTANTS import EPOCHS_CNN, MOMENTUM, LEARNING_RATE_CNN
from cnn import CNNModel
from utils import load_dataset

def train_cnn(model, trainloader, testloader, device, epochs = EPOCHS_CNN, lr=LEARNING_RATE_CNN, momentum=MOMENTUM):
    """
    Trains the CNN model

    Args:
        model (torch.nn.Module): The model to be trained.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to perform computations on.
        epochs (int, optional): Number of epochs to train the model. Defaults to EPOCHS_CNN.
        lr (float, optional): Learning rate for the optimizer. Defaults to LEARNING_RATE_CNN.
        momentum (float, optional): Momentum factor for the SGD optimizer. Defaults to MOMENTUM.

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

def get_trained_CNN_model(trainloader, testloader, device, weights_file_path = './models/trained_model.pth'):
    """
    Load the trained CNN model if weights file exists, else train the model on the dataset.
    
    Args:
        trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to train the model on.
        weights_file_path (str, optional): Path to the trained model's weights. Default is './models/trained_model.pth'.

    Returns:
        model (torch.nn.Module): The trained CNN model.
        accuracy (float): Returns the test accuracy of the trained model.
    """
    if os.path.isfile(weights_file_path):
        print(f"Loading pre-trained model from {weights_file_path}")

        model = CNNModel()
        model.load_state_dict(torch.load(weights_file_path, map_location=device))
        model = model.to(device)

        # Evaluate the trained model on the test set
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
        print(f"Model Accuracy on Test Set - {accuracy}%")
        print("="*50)
    else:
        print(f"Training a new model and saving it to {weights_file_path}")
        model = CNNModel().to(device)
        accuracy = train_cnn(model, trainloader, testloader, device) # Model trained inplace
        torch.save(model.state_dict(), weights_file_path)
    return model, accuracy