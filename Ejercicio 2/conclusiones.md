# Conclusiones

## Ingeniería de características: Discretización del estado

Para entrenar al agente Q-learning en el entorno de Flappy Bird, se realizó una discretización manual del estado del juego. Se seleccionaron cinco variables como representación del entorno:

1. Distancia horizontal al primer tubo (`next_pipe_dist_to_player`)
2. Diferencia vertical entre el jugador y el centro del hueco del primer tubo
3. Distancia horizontal al segundo tubo (`next_next_pipe_dist_to_player`)
4. Diferencia vertical entre el jugador y el centro del hueco del segundo tubo
5. Velocidad vertical del jugador (`player_vel`)

Estas variables fueron discretizadas mediante binning:

- `DX1_BIN = 25` (primer tubo)
- `DX2_BIN = 40` (segundo tubo)
- `DY_BIN = 25` (diferencia vertical)
- La velocidad del jugador se mantuvo como un entero ya discreto.

Esta estrategia buscó evitar la explosión del espacio de estados, optando por una discretización no demasiado fina. La elección funcionó correctamente: permitió que el agente generalice sin requerir una cantidad excesiva de episodios para cubrir los posibles estados.

---

## Análisis y comparación de agentes

### Agente Q-Learning

- **Episodios de entrenamiento**: 20.000
- **Parámetros**:
  - Learning rate: 0.2
  - Discount factor: 0.95
  - Epsilon inicial: 1.0, mínimo: 0.05, decay: 0.995
- **Recompensa promedio final**: entre 15 y 20 puntos
- **Generalización**: buena. El agente mantuvo su performance al ejecutar en test sin exploración (`epsilon = 0`).
- **Ventajas**:
  - Implementación simple
  - Resultados estables
- **Limitaciones**:
  - Espacio de estados limitado por la discretización
  - La tabla puede crecer si se intenta una discretización más fina

---

### Agente con red neuronal (DQN con Q-table aproximada)

- **Arquitectura del modelo**:
```python
model = keras.Sequential([
     layers.Input(shape=(X.shape[1],)), # (shape=(5,))  
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(y.shape[1]) # (dos acciones posibles)
])  
```

- **Entrenamiento**:
  - Epochs: 90
  - Batch size: 64
  - Validation split: 0.2
  - Optimizer: Adam
  - Loss: MSE

- **Resultados**:
  - El modelo convergió sin overfitting
  - Las curvas de `loss` y `val_loss` evolucionaron de forma conjunta y convergieron en torno a 0.15
  - En test, las recompensas obtenidas variaron entre -1 y 159, lo cual indica que puede generalizar y aprender políticas útiles
  - Se probó con distintos parámetros de ejecución (por ejemplo, usando `frame_skip=2`) y el agente mantuvo su rendimiento, lo que refuerza la robustez de la aproximación neuronal

- **Gráfico del entrenamiento**:

![Evolución del error durante el entrenamiento](Ejercicio 2\curva-loss.jpg)

- **Ventajas**:
  - Permite generalizar a estados no vistos
  - Reduce el tamaño necesario de memoria frente a una Q-table explícita
  - Es un paso hacia DQN completo, con capacidad de extender a estados continuos sin discretización

- **Desventajas**:
  - Mayor complejidad computacional
  - La predicción con redes neuronales es más lenta que acceder a una tabla, especialmente visible en la ejecución frame a frame del juego

---

## Reflexiones finales

Ambos enfoques lograron aprender a jugar de forma competente al Flappy Bird. El Q-learning clásico es eficaz con una buena ingeniería de características y discretización, pero escala mal para estados más complejos o continuos. Por su parte, el modelo basado en red neuronal mostró buena capacidad de generalización y es más flexible, aunque con un mayor coste computacional.

Este trabajo sienta las bases para evolucionar hacia **Deep Q-Learning (DQN)**, donde ya no sería necesario discretizar los estados
