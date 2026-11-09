from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Charger MNIST (téléchargé automatiquement la 1re fois)
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# étape de prétraitement
X_train = X_train /255
X_test = X_test /255 


modele = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # aplatit la grille 28x28 en 784 entrées
    layers.Dense(128, activation="relu"),    # couche cachée de 128 neurones
    layers.Dense(64,  activation="relu"),    # couche cachée de 64 neurones
    layers.Dense(64,  activation="relu"),    # couche cachée de 64 neurones
    layers.Dense(32,  activation="relu"),    # couche cachée de 32 neurones
    layers.Dense(16,  activation="relu"),    # couche cachée de 16 neurones
    layers.Dense(10,  activation="softmax")  # 10 neurones de sortie (un par chiffre 0-9)
])

# Configurer l'entraînement
modele.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# Entraîner : 5 passages (epochs) sur les données
historique = modele.fit(X_train, y_train,
                        epochs=15,
                        validation_split=0.1)   # garde 10% pour surveiller l'overfitting

# Évaluer sur les données de test (jamais vues)
perte, precision = modele.evaluate(X_test, y_test)
print("Précision sur le test :", precision)     # typiquement ~0.97  ->  97 % !

# Prédictions sur le jeu de test
predictions = modele.predict(X_test)
chiffres_predits = np.argmax(predictions, axis=1)   # le chiffre le plus probable

# Trouver les indices où le réseau s'est trompé
erreurs = np.where(chiffres_predits != y_test)[0]
print("Nombre d'erreurs :", len(erreurs))

# Afficher l'architecture
modele.summary()
i = erreurs[0]
plt.imshow(X_test[i], cmap="gray")
plt.title("Réel : " + str(y_test[i]) + "  |  Prédit : " + str(chiffres_predits[i]))
plt.show()