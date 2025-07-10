import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras

# === Cargar dataset JSON ===
with open('dataset_manos.json') as f:
    data = json.load(f)

X = np.array([d['landmarks'] for d in data])
y_text = [d['label'] for d in data]

# === Codificar etiquetas ===
le = LabelEncoder()
y = le.fit_transform(y_text)

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y_text, test_size=0.2)

# === Modelo ===
model = keras.Sequential([
    keras.layers.Input(shape=(63,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Entrenamiento ===
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# === Guardar el modelo en HDF5 ===
model.save("modelo_manos.h5")
print("✅ Modelo entrenado y guardado como modelo_manos.h5")

# === Guardar etiquetas para usar luego en la web ===
np.save("labels.npy", le.classes_)
print("✅ Etiquetas guardadas en labels.npy")
