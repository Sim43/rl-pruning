import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=1, padding=1)
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
        self.fc3 = nn.Linear(6272, 512)

        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 1x28x28, output 64x28x28
        x = self.drop1(self.act1(self.conv1(x)))

        # input 64x28x28, output 128x28x28
        x = self.drop2(self.act2(self.conv2(x)))

        # input 128x28x28, output 64x28x28
        x = self.drop3(self.act3(self.conv3(x)))

        # input 64x28x28, output 32x28x28
        x = self.drop4(self.act4(self.conv4(x)))

        # input 32x28x28, output 32x14x14
        x = self.pool(x)

        # input 32x14x14, output 6272
        x = self.flat(x)
        x = self.act5(x)
        x = self.drop5(x)

        # input 6272, output 512
        x = self.fc3(x)

        # input 512, output 10
        x = self.fc4(x)
        return x