import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(4*4*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = tuple(str(i) for i in range(10))

    loss_fn = torch.nn.CrossEntropyLoss()

    net = SimpleConvNet()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    losses = []

    for epoch in tqdm_notebook(range(3)):

        running_loss = 0.0
        for i, batch in enumerate(tqdm_notebook(trainloader)):
            x_batch, y_batch = batch

            optimizer.zero_grad()

            y_pred = net(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i+1, running_loss / 2000))
                losses.append(running_loss)
                running_loss = 0.0

    class_correct = list(0. for i in range(10))
    class_total = list(0.for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            y_pred = net(images)
            _, predicted = torch.max(y_pred, 1)
            c = (predicted == labels)

            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


