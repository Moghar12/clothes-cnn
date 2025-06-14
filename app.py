import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import FashionMNISTModel

# Classes du dataset Fashion MNIST
CLASSES = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)

# Prétraitement identique à l'entraînement
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Charger le modèle
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTModel().to(device)
    model.load_state_dict(torch.load('fashion_mnist_cnn.pth', map_location=device))
    model.eval()
    return model, device

# Fonction de prédiction
def predict(image, model, device):
    image = transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return CLASSES[predicted.item()]

# Interface Streamlit
st.title("👕 Classificateur Fashion MNIST")

uploaded_file = st.file_uploader("📤 Uploade une image de vêtement (T-shirt, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Image chargée", use_column_width=True)

    model, device = load_model()
    prediction = predict(image, model, device)

    st.success(f"✅ Classe prédite : **{prediction}**")
