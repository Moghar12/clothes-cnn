import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


def load_mnist_data(path, kind='train'):
    """
    Charge les données MNIST/Fashion-MNIST à partir de fichiers .ubyte
    """
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)  # En-tête de 8 octets
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)  # En-tête de 16 octets
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 28, 28)

    return images, labels


class FashionMNISTCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], mode='L')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_fashion_mnist_data(data_dir='data', batch_size=64):
    print(f"Chargement des données depuis {data_dir}...")

    train_images, train_labels = load_mnist_data(data_dir, 'train')
    test_images, test_labels = load_mnist_data(data_dir, 't10k')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = FashionMNISTCustomDataset(train_images, train_labels, transform=transform)
    testset = FashionMNISTCustomDataset(test_images, test_labels, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    return trainloader, testloader, classes


if __name__ == '__main__':
    trainloader, testloader, classes = get_fashion_mnist_data('data')

    print(f"{len(trainloader)} batches d'entraînement")
    print(f"{len(testloader)} batches de test")

    images, labels = next(iter(trainloader))
    print(f"Image batch shape : {images.shape}")
    print(f"Premier label : {classes[labels[0]]}")
