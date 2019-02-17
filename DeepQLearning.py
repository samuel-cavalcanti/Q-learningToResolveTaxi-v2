from keras import layers
from keras import models
from keras.optimizers import Adam
from Qlearning import Q_learning
import numpy as np
from collections import deque
import random


class DeepQLearning(Q_learning):
    # reference https://keon.io/deep-q-learning/
    learning_rate = 0.001
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 40

    def __init__(self, size_of_state, max_num_actions, random_func, alpha=None, gamma=None, epsilon=None,
                 learning_rate=None, epsilon_min=None, epsilon_decay=None):

        self.random = random_func
        self._setHyperParameters(alpha, gamma, epsilon, learning_rate, epsilon_min, epsilon_decay)
        self._buildModel(size_of_state, max_num_actions)
        self.memory = deque(maxlen=500)

    def _buildModel(self, input_size, output_size):

        input_layer = layers.Input(shape=(input_size,))
        first_hidden_layer = layers.Dense(60, activation="relu")(input_layer)
        second_hidden_layer = layers.Dense(60, activation="relu")(first_hidden_layer)
        output_layer = layers.Dense(output_size, activation="linear")(second_hidden_layer)
        self.model = models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss="mse")

    def _setHyperParameters(self, alpha, gamma, epsilon, learning_rate, epsilon_min, epsilon_decay):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if epsilon_min is not None:
            self.epsilon_min = epsilon_min
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay

        super()._setHyperParameters(alpha, gamma, epsilon)

    def _remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def _toInputarray(self, state):
        input_array = []

        for i in range(len(state)):
            for j in range(len(state[i][0])):
                if j == 3:
                    input_array.append(int(state[i][0][j] == True))
                else:
                    input_array.append(state[i][0][j])

        # print(input_array)
        return np.array(input_array).reshape((1, -1))

    def _predictModel(self, state):
        input_array = self._toInputarray(state)
        return self.model.predict(input_array)

    def _explorationOrExploitation(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.random()
        else:
            predict = self._predictModel(state)
            action = np.argmax(predict)

        return action

    def execute(self, env):
        current_state = env.P[env.s]
        action = self._explorationOrExploitation(current_state)

        next_state_id, reward, done, info = env.step(action)

        next_state = env.P[next_state_id]

        self._remember(current_state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            self._replay()

        return action

    # https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def _replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        array_states, array_nodes = [], []

        for current_state, action, reward, next_state, done in mini_batch:
            new_value = reward + self.gamma * np.amax(self._predictModel(next_state))
            nodes = self._predictModel(current_state)
            nodes[0][action] = new_value
            array_states.append(self._toInputarray(current_state)[0])
            array_nodes.append(nodes[0])

        history = self.model.fit(np.array(array_states), np.array(array_nodes), epochs=1, verbose=0).history
        loss = history["loss"][0]

        return loss

    # def _update(self, current_state, next_state, action, reward):
    #     nodes = self._predictModel(current_state)
    #
    #     new_value = reward + self.gamma * np.amax(self._predictModel(next_state))
    #
    #     nodes[0][action] = new_value
    #
    #     self.model.fit(self._toInputarray(current_state), nodes, epochs=1, verbose=0)
    #
    #     # não sei se coloco
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def load(self, file):
        models.load_model(file)

    def save(self, file):
        self.model.save(file)
