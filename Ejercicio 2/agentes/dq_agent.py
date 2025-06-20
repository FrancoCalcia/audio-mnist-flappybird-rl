from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle
import random 

class QAgent(Agent):
    """
    Agente de Q-Learning.
    Completar la discretización del estado y la función de acción.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.99,
                 epsilon=0.0, epsilon_decay=0.995, min_epsilon=0.05, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        
       
        self.DX1_BIN = 25   # primer tubo: más fino
        self.DX2_BIN = 40   # segundo tubo: más grueso
        self.DY_BIN  = 25


    def discretize_state(self, state):
        # Distancia + diferencia vertical al 1.º tubo
        dx1 = state["next_pipe_dist_to_player"]
        gap1_center = (state["next_pipe_top_y"] + state["next_pipe_bottom_y"]) / 2
        dy1 = state["player_y"] - gap1_center

        # Distancia + diferencia vertical al 2.º tubo
        dx2 = state["next_next_pipe_dist_to_player"]
        gap2_center = (state["next_next_pipe_top_y"] + state["next_next_pipe_bottom_y"]) / 2
        dy2 = state["player_y"] - gap2_center

        vel = state["player_vel"]

        # Binning
        dx1_bin = int(dx1 // self.DX1_BIN)
        dx2_bin = int(dx2 // self.DX2_BIN)
        dy1_bin = int(dy1 // self.DY_BIN)
        dy2_bin = int(dy2 // self.DY_BIN)
        vel_bin = int(vel)          # ya discreto

        return (dx1_bin, dy1_bin, dx2_bin, dy2_bin, vel_bin)


    def act(self, state):
        """
        Política ε-greedy sobre la Q-table:
            • Con probabilidad ε → acción aleatoria.
            • Si no → argmax de Q(s,a).
        """
        discrete_state = self.discretize_state(state)

        # Asegurarnos de que el estado exista en la tabla
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))

        # Explorar
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Explotar
        best_action_idx = int(np.argmax(self.q_table[discrete_state]))
        return self.actions[best_action_idx]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de Q-learning.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
