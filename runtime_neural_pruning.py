import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import CONSTANTS

from decision_network import train_decision_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONSTANTS.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=CONSTANTS.batch_size, shuffle=True)

def train_cnn(model, i):
    epochs = CONSTANTS.num_epochs_finetune
    if i == CONSTANTS.num_CNN_layers - 2:
        epochs *= 2

    loss_fn = nn.CrossEntropyLoss()
    optimizer_cnn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward, backward, and then weight update
            y_pred = model(inputs)
            loss_cnn = loss_fn(y_pred, labels)

            optimizer_cnn.zero_grad()
            loss_cnn.backward()
            optimizer_cnn.step()
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

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 50, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(50, 20, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(20, 12, kernel_size=(3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()

        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(3072, 512)
        self.act5 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        
        # input 32x32x32, output 50x32x32
        x = self.act2(self.conv2(x))

        # input 50x32x32, output 20x32x32
        x = self.act3(self.conv3(x))

        # input 20x32x32, output 12x32x32
        x = self.act4(self.conv4(x))

        # input 12x32x32, output 12x16x16
        x = self.pool4(x)

        # input 12x16x16, output 3072
        x = self.flat(x)
        # input 4096, output 512
        x = self.act5(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)
        return x
    
model = CIFAR10Model()
model.load_state_dict(torch.load('/home/hice1/rdesai316/scratch/project/models/cifar10model.pth', map_location=device))
model = model.to(device)

# # step 1 done (initialize: train C in normal way or initialize C with pre-trained model)
return_nodes = ['drop1', 'drop2', 'drop3', 'drop4']
conv_layers = ['conv1', 'conv2', 'conv3','conv4']

multiplications = 0
for i in range(len(conv_layers)):
    output_img_size = 32 if i != len(conv_layers) - 1 else 16

    multiplications += getattr(model, conv_layers[i]).kernel_size[0]\
                       *getattr(model, conv_layers[i]).kernel_size[1]\
                       *getattr(model, conv_layers[i]).in_channels\
                       *getattr(model, conv_layers[i]).out_channels\
                       *(output_img_size**2)

print("Total multiplications for full model for one batch: ", multiplications)

output_img_size = 32
multiplications = getattr(model, conv_layers[0]).kernel_size[0]\
                *getattr(model, conv_layers[0]).kernel_size[1] \
                *getattr(model, conv_layers[0]).in_channels\
                *getattr(model, conv_layers[0]).out_channels\
                *(output_img_size**2)

for i in range(CONSTANTS.num_CNN_layers - 1):    
    multiplications += train_decision_network.driver(trainloader, model, i, return_nodes, conv_layers)
    train_cnn(model, i)

print("Total multiplications using RNP framework: ", multiplications)
