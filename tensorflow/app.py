import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Classes Fashion MNIST
CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Chargement du modèle Keras
@st.cache_resource
def load_keras_model():
    model = load_model('mnist-fashion-model.h5')
    return model

# Fonction de prédiction
def predict(image, model):
    image = image.convert('L').resize((28, 28))  # Niveaux de gris + redimension
    img_array = img_to_array(image) / 255.0      # Normalisation [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    return CLASSES[class_index]

# Streamlit interface
st.title("👕 Classificateur Fashion MNIST")

uploaded_file = st.file_uploader("📤 Uploade une image de vêtement", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Image chargée", use_column_width=True)
    
    model = load_keras_model()
    prediction = predict(image, model)
    st.success(f"✅ Classe prédite : **{prediction}**")
