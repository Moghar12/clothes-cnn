import torch
import torch.nn as nn
import torch.optim as optim
from model import FashionMNISTModel
from data_loader import get_fashion_mnist_data

def train_model(epochs=10, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    trainloader, testloader, classes = get_fashion_mnist_data('data', batch_size)
    model = FashionMNISTModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'fashion_mnist_cnn.pth')
    print("Modèle sauvegardé sous fashion_mnist_cnn.pth")

if __name__ == '__main__':
    train_model()
