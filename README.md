# 👕 Classificateur Fashion MNIST avec Streamlit

Cette application permet de classifier des images de vêtements à l'aide d'un modèle CNN entraîné sur le dataset **Fashion MNIST**. Elle utilise **Streamlit** pour proposer une interface simple et interactive où tu peux uploader une image de vêtement et obtenir la prédiction du modèle.

---

## 📦 Fichiers principaux

- `app.py` : Interface utilisateur Streamlit.
- `model.py` : Définition du modèle CNN `FashionMNISTModel`.
- `fashion_mnist_cnn.pth` : Poids du modèle entraîné (fichier requis).
- `README.md` : Documentation du projet.

---

## 🧠 Classes du dataset Fashion MNIST

Le modèle prédit l'une des 10 classes suivantes :

1. T-shirt/top  
2. Trouser  
3. Pullover  
4. Dress  
5. Coat  
6. Sandal  
7. Shirt  
8. Sneaker  
9. Bag  
10. Ankle boot  

---

## ⚙️ Installation

Assure-toi que Python 3.7 ou supérieur est installé, puis installe les dépendances :

```bash
pip install streamlit torch torchvision pillow
```

---

## 🚀 Lancer l’application

Utilise la commande suivante pour démarrer l’application Streamlit :

```bash
streamlit run app.py
```

---

## 🧪 Utilisation

1. Lance l’application avec streamlit run app.py.

2. Dans l’interface Streamlit, clique sur "Uploade une image de vêtement".

3. Choisis une image au format .png, .jpg, ou .jpeg.

4. L’image est automatiquement prétraitée et envoyée au modèle.

5. La classe prédite s’affiche en bas.

---

## 🖼️ Interface utilisateur

<img width="424" alt="Capture d’écran 2025-06-14 à 12 48 13" src="https://github.com/user-attachments/assets/a9f3d7f2-8784-4ab4-82b7-3eefc29018b9" />

---

## 📬 Contact

Développé par Ali Moghar
📧 Email : mogharali10@gmail.com

