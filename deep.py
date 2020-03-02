import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random

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

# Classes, G0 stands for normal diagnosis.
classes = (
    "G0",
    "G1",
    "G2",
    "G3",
)


class Net(nn.Module):
    """
    Classic PyTorch CNN model for image classification.
    See: https://pytorch.org/tutorials/
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 122 * 122, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 4, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 122 * 122)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Initial population size for every epoch
POP_SIZE = 10

# Always odd value!
DEFAULT_ELITE_SIZE = 6


def gen_population(shape, size=POP_SIZE):
    res = []
    for i in range(size):
        res.append(torch.randn(shape))
    return res


def get_fittest(list_of_cand, elite_size=DEFAULT_ELITE_SIZE):
    """
    List of caditates is always sorted by first value of
    tuple (loss value) in ASC order
    """
    selected = []
    for i in range(elite_size):
        selected.append(list_of_cand[i][1])

    selection_size = len(list_of_cand)

    for i in range(0, selection_size - elite_size):
        pick_value = random.uniform(0, list_of_cand[selection_size - 1][0])
        for j in range(0, selection_size):
            if list_of_cand[i][0] <= pick_value:
                selected.append(list_of_cand[i][1])
                break

    diff = selection_size - len(selected)

    for i in range(0, diff):
        ran_i = random.randint(0, selection_size - 1)
        selected.append(list_of_cand[ran_i][1])

    return selected


def cross_by_two(p1, p2):
    """
    One-Point Crossover:
    One random point is chosen on the individual chromosomes
    and the genetic material is exchanged at these point.
    """
    p_size = list(p1.size())[0]

    k = random.randint(1, p_size - 1)
    d1, d2 = p1, p2
    for i in range(k, p_size):
        d1[i], d2[i] = d2[i], d1[i] 

    return d1, d2


def do_crossover(chromozomes):
    """
    Do Two-Point Crossover for selection on current epoch.
    """
    next_gen = []

    for i in range(0, len(chromozomes) - 1, 2):
        parent_1 = chromozomes[i]
        parent_2 = chromozomes[i + 1]
        des1, des2 = cross_by_two(parent_1, parent_2)
        next_gen.append(des1)
        next_gen.append(des2)

    return next_gen


def do_mutation():
    pass


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Salvez acuratetea si lossul pe testing dupa ce iau cel rezultat
# Amestecand datele
# Probabilitatea de mutatie setata de mine.
# losst
# Use normalized loss -1 ---- 1 to mutate random weigth using mutation algo.
def run():
    # Shuffle dataset
    for epoch in range(1):  # loop over the dataset multiple times
        with torch.no_grad():
            # print(net.conv1.weight)
            # print(net.fc1.weight)
            # print(net.fc2.weight.shape)
            # print(net.fc3.weight.shape)

            # t = net.conv1.weight
            # t = t.view(np.prod(t.shape))
            # print(t)
            # t = t.view(net.conv1.weight.shape)
            # print(t)

            conv1_pop = gen_population(net.conv1.weight.shape)
            # print(conv1_pop)

            # run other data generated population. adjust weights, select the best boi in every epoch
            # add the best boio to final pop
            # select from final pop boio with best fitness
            # check result
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                cv1_fit = []
                # Get fitness for gen
                for i in range(POP_SIZE):

                    net.conv1.weight = torch.nn.Parameter(conv1_pop[i])

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    # print(loss)
                    cv1_fit.append((loss.item(), conv1_pop[i].view(
                        np.prod(net.conv1.weight.shape))))

                cv1_fit = sorted(cv1_fit, key=lambda x: x[0])
                cv1_selected = get_fittest(cv1_fit)
                print(len(cv1_selected))
                do_crossover(cv1_selected)
    print('Finished Training')


if __name__ == '__main__':
    run()
