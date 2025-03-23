import random
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from CONSTANTS import SEED, BATCH_SIZE
from torch import cuda, device, manual_seed, nn
from typing import Optional, Tuple

def load_dataset(dataset_name: str = "FashionMNIST", root='./data') -> Tuple[DataLoader, DataLoader]:
    """
    Load dataset and create DataLoader objects for training and testing.

    Args:
        dataset (str): Dataset to load. Default is FashionMNIST.
        root (str): Root directory where the dataset will be stored. Default is './data'.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing:
            - trainloader: DataLoader for the training set
            - testloader: DataLoader for the test set
    """
    if dataset_name in ["FashionMNIST", "CIFAR10"]:
        dataset_class = getattr(datasets, dataset_name)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    
    transform = transforms.Compose([transforms.ToTensor()])  # Convert images to PyTorch tensors

    trainset = dataset_class(root=root, train=True, download=True, transform=transform)
    testset = dataset_class(root=root, train=False, download=True, transform=transform)

    # Create separate DataLoaders for training and testing sets
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"Dataset selected: {dataset_name}.\nThe train set contains {len(trainloader)} batches and test set contains {len(testloader)} batches")
    return trainloader, testloader

def get_device() -> device:
    """
    Get the device to be used (GPU if available, otherwise CPU).

    Returns:
        device (torch.device): The device to be used.
    """
    if cuda.is_available():
        selected_device = device("cuda:0")
        print("Using GPU for training")
    else:
        selected_device = device("cpu")
        print("Using CPU for training. Please consider using a GPU for faster training.")
    return selected_device

def set_random_seed(seed = SEED) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set. Default is SEED from CONSTANTS.
    """
    manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Random seed set to {seed}")

def get_multiplications_per_conv_layer(conv_layer: nn.Conv2d, output_img_size: int, in_channels: Optional[int] = None) -> int:
    """
    Calculate the number of multiplications per image in a given convolutional layer.

    Args:
        conv_layer (torch.nn.Conv2d): The convolutional layer.
        output_img_size (int, optional): The height/width of the output feature map. Defaults to 32.

    Returns:
        int: The total number of multiplications performed in the given convolutional layer.
    """
    if in_channels == None:
        in_channels = conv_layer.in_channels
    return conv_layer.kernel_size[0] * conv_layer.kernel_size[1] * in_channels * conv_layer.out_channels * (output_img_size ** 2)