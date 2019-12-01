from torchvision import datasets, transforms
import torch



def load_train_dataset():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    dataset_path = 'images/train/'
    train_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True,
    )
    return train_loader



def load_test_dataset():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    dataset_path = 'images/test/'
    test_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
    )
    return test_loader
