from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def act(self, state):
        """
        COMPLETAR: Implementar la función de acción.
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        raise NotImplementedError("Completar la función act para NNAgent")
