import torch
from torch import nn, optim

from load_datasets import (
    load_train_dataset,
    load_test_dataset,
)
from net_model import Net


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



# Train the Model with Train Image Datasets and Saves cifar_net.pth file

def train_model():
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(load_train_dataset(), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # print('inputs: ', inputs)
            # print('labels: ', labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), 'images/cifar_net.pth')

    return True
