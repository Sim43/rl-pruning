import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import CONSTANTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONSTANTS.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=CONSTANTS.batch_size, shuffle=True)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)


        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.1)

        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(128, 100, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.1)

        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv4 = nn.Conv2d(100, 64, kernel_size=(3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.1)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.1)


        self.conv6 = nn.Conv2d(32, 25, kernel_size=(3,3), stride=1, padding=1)
        self.act6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.1)

        self.conv7 = nn.Conv2d(25, 20, kernel_size=(3,3), stride=1, padding=1)
        self.act7 = nn.ReLU()
        self.drop7 = nn.Dropout(0.1)

        self.conv8 = nn.Conv2d(20, 16, kernel_size=(3,3), stride=1, padding=1)
        self.act8 = nn.ReLU()
        self.drop8 = nn.Dropout(0.1)

        self.pool6 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(4096, 512)
        self.act5 = nn.ReLU()
        self.drop7 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 64x32x32
        x = self.drop1(self.act1(self.conv1(x)))
        
        # input 64x32x32, output 128
        x = self.drop2(self.act2(self.conv2(x)))

        # input 128x32x32, output 100x32x32
        x = self.drop3(self.act3(self.conv3(x)))

        # input 100x32x32, output 64x32x32
        x = self.drop4(self.act4(self.conv4(x)))

        # input 64x32x32, output 32x32x32
        x = self.drop5(self.act5(self.conv5(x)))

        # input 32x32x32, output 25x32x32
        x = self.drop6(self.act6(self.conv6(x)))

        # input 25x32x32, output 20x32x32
        x = self.drop7(self.act7(self.conv7(x)))

        # input 20x32x32, output 16x32x32
        x = self.drop8(self.act8(self.conv8(x)))


        # input 16x32x32, output 16x16x16
        x = self.pool6(x)

        # input 16x16x16, output 4096
        x = self.flat(x)
        # input 4096, output 512
        x = self.act5(self.fc3(x))
        x = self.drop7(x)
        # input 512, output 10
        x = self.fc4(x)
        return x

model = CIFAR10Model().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    for inputs, labels in tqdm(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))

torch.save(model.state_dict(), "/home/hice1/rdesai316/scratch/project/models/cifar10model.pth")