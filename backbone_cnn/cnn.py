import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for image classification.
    """
    def __init__(self, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)


        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.1)
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()
        self.fc3 = nn.Linear(32*(img_size//2)*(img_size//2), 512)

        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, in_channels, img_size, img_size)
        x = self.drop1(self.act1(self.conv1(x)))

        # input: (batch_size, 64, img_size, img_size)
        x = self.drop2(self.act2(self.conv2(x)))

        # input: (batch_size, 128, img_size, img_size)
        x = self.drop3(self.act3(self.conv3(x)))

        # input: (batch_size, 64, img_size, img_size)
        x = self.drop4(self.act4(self.conv4(x)))

        # input: (batch_size, 32, img_size, img_size)
        x = self.pool(x)

        # input: (batch_size, 32, img_size//2, img_size//2)
        x = self.flat(x)
        x = self.act5(x)
        x = self.drop5(x)

        # input: (batch_size, 32 * (img_size//2) * (img_size//2))
        x = self.fc3(x)

        # input: (batch_size, 512)
        x = self.fc4(x)
        # output: (batch_size, 10)
        return x