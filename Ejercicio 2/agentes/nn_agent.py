from agentes.base import Agent
import numpy as np
import tensorflow as tf

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

    def discretize_state(self, state):
        DX1_BIN = 25
        DX2_BIN = 40
        DY_BIN = 25

        dx1 = state["next_pipe_dist_to_player"]
        gap1_center = (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2
        dy1 = state["player_y"] - gap1_center

        dx2 = state["next_next_pipe_dist_to_player"]
        gap2_center = (state["next_next_pipe_top_y"] + state["next_next_pipe_bottom_y"]) / 2
        dy2 = state["player_y"] - gap2_center

        vel = state["player_vel"]

        dx1_bin = int(dx1 // DX1_BIN)
        dx2_bin = int(dx2 // DX2_BIN)
        dy1_bin = int(dy1 // DY_BIN)
        dy2_bin = int(dy2 // DY_BIN)
        vel_bin = int(vel)

        return np.array([dx1_bin, dy1_bin, dx2_bin, dy2_bin, vel_bin], dtype=np.float32)

    def act(self, state):
        """
        Usa el modelo para predecir la mejor acción desde el estado actual.
        """
        state_array = self.discretize_state(state).reshape(1, -1)

        #predecir Q-values para cada acción
        #q_values = self.model.predict(state_array, verbose=0)[0]  #[0]
        q_values = self.model(state_array, training=False).numpy()[0]

        #acción con mayor valor Q
        best_action_index = np.argmax(q_values)

        #devolver acción correspondiente del espacio de acciones
        return self.actions[best_action_index]
