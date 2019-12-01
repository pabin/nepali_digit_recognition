import torch
from load_datasets import load_test_dataset
from net_model import Net

net = Net()

def calculate_accuracy():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in load_test_dataset():
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nTotal: ', total)
    print('Correct: ', correct)

    print('\nAccuracy of the network on the %d test images: %d %%' % (total,
        100 * correct / total))

    return True
