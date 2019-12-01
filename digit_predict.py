import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from net_model import Net
from load_datasets import load_test_dataset
from calculate_accuracy import calculate_accuracy
from train_model import train_model

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
net = Net()

#Calls Train Model
train_model()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(load_test_dataset())
images, labels = dataiter.next()

test_batch_size = 1
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(test_batch_size)))
# imshow(torchvision.utils.make_grid(images))

net.load_state_dict(torch.load('images/cifar_net.pth'))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(test_batch_size)))

# Calls Accuracy Calculation Function
calculate_accuracy()




# dataiter = iter(load_image_dataset())
# images, labels = dataiter.next()
#
#
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(64)))
#
# # show images
# imshow(torchvision.utils.make_grid(images))
