import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl' 
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
X = np.array(X)
y = np.array(y)

# --- Definir la red neuronal ---
model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),   
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(y.shape[1])  #valores Q (uno por acción)
])

model.compile(optimizer='adam', loss='mse')

# --- Entrenar la red neuronal ---
history = model.fit(X, y, epochs=90, batch_size=64, verbose=1, validation_split=0.2)

# --- Mostrar resultados del entrenamiento ---
plt.plot(history.history['loss'], label='loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Evolución del error durante el entrenamiento")
plt.show()

# --- Guardar el modelo entrenado ---
model.save('flappy_q_nn_model.keras')
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')


# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
