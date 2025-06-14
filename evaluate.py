import torch
from model import FashionMNISTModel
from data_loader import get_fashion_mnist_data

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation avec : {device}")

    _, testloader, classes = get_fashion_mnist_data('data', batch_size=64)

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

    print(f"Accuracy du modèle sur le test set : {100 * correct / total:.2f}%")

if __name__ == '__main__':
    evaluate_model()
