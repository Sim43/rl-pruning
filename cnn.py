import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1)
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
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(8192, 512)
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 64x32x32
        x = self.drop1(self.act1(self.conv1(x)))
        
        # input 64x32x32, output 128x32x32
        x = self.drop2(self.act2(self.conv2(x)))

        # input 128x32x32, output 64x32x32
        x = self.drop3(self.act3(self.conv3(x)))

        # input 64x32x32, output 32x32x32
        x = self.drop4(self.act4(self.conv4(x)))

        # input 32x32x32, output 32x16x16
        x = self.pool(x)

        # input 32x16x16, output 8192
        x = self.flat(x)
        # input 8192, output 512
        x = self.act5(self.fc3(x))
        x = self.drop5(x)
        # input 512, output 10
        x = self.fc4(x)
        return x