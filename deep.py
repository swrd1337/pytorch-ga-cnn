import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# numpy train test split NFoldCrossValidation
transform = transforms.Compose(
    [transforms.Resize((500, 500)), transforms.ToTensor()]
)

trainset = torchvision.datasets.ImageFolder(
    root="./data", transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=True, num_workers=2
)

testset = torchvision.datasets.ImageFolder(
    root="./test", transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=False, num_workers=2
)

classes = (
    "G0",
    "G1",
    "G2",
    "G3",
)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 122 * 122, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 122 * 122)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def gen_population(shape, size=10):
    res = []
    for i in range(size):
        #WEIHTS NORMALIZATION
        # intree ca -1 si 1
        # see net.conv1.weight
        res.append(torch.randn(shape))
    return res

# Salvez acuratetea si lossul pe testing dupa ce iau cel rezultat
# Amestecand datele
# Probabilitatea de mutatie setata de mine.
# losst
def run():
    # Shuffle dataset 
    for epoch in range(1):  # loop over the dataset multiple times

        conv1_pop = gen_population(net.conv1.weight.shape)
        conv2_pop = gen_population(net.conv2.weight.shape)

        # run other data generated population. adjust weights, select the best boi in every epoch
        # add the best boio to final pop
        # select from final pop boio with best fitness
        # check result
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            cv1_fit = {}
            cv2_fit = {}
            # Get fitness for gen
            for i in range(10):
                with torch.no_grad():
                    net.conv1.weight = torch.nn.Parameter(conv1_pop[i])
                    net.conv2.weight = torch.nn.Parameter(conv2_pop[i])
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                cv1_fit[loss.item()] = conv1_pop[i]
                cv2_fit[loss.item()] = conv2_pop[i]

            # cv1_fit = sorted(cv1_fit)
            print(cv1_fit)

            # loss.backward()
            # optimizer.step()

            # running_loss += loss.item()
            # print('[%d, %5d] loss: %.3f' %
            #     (epoch + 1, i + 1, running_loss))
            # running_loss = 0.0
    print('Finished Training')
    

if __name__ == '__main__':
    # pass
    run()
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html