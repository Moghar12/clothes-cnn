import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision  # ✅ correction du NameError
from train import train_model
from evaluate import evaluate_model
from model import FashionMNISTModel
from data_loader import get_fashion_mnist_data

def imshow(img):
    img = img / 2 + 0.5  # dénormaliser
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.show()

def show_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader, classes = get_fashion_mnist_data('data', batch_size=8)

    model = FashionMNISTModel().to(device)
    model.load_state_dict(torch.load('fashion_mnist_cnn.pth', map_location=device))
    model.eval()

    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print("🎯 Étiquettes réelles :", ' | '.join(f'{classes[l]}' for l in labels))
    print("🤖 Prédictions       :", ' | '.join(f'{classes[p]}' for p in predicted))

    imshow(torchvision.utils.make_grid(images.cpu()))

def evaluate_and_return_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, testloader, _ = get_fashion_mnist_data('data', batch_size=64)

    model = FashionMNISTModel().to(device)
    model.load_state_dict(torch.load('fashion_mnist_cnn.pth', map_location=device))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    print("📦 Entraînement du modèle...")
    train_model(epochs=10)

    print("\n🔍 Évaluation du modèle...")
    acc = evaluate_and_return_accuracy()
    print(f"✅ Accuracy du modèle sur le test set : {acc:.2f}%")

    print("\n🖼️ Affichage des prédictions sur un batch test...")
    show_predictions()
