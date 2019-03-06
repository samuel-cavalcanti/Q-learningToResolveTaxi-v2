from keras import layers
from keras import models
from keras.optimizers import Adam
from Qlearning import Q_learning
import numpy as np
from collections import deque
import random


class DeepQLearning(Q_learning):
    # reference https://keon.io/deep-q-learning/
    _learning_rate = 0.001
   
    _batch_size = 40

    def __init__(self, size_input_layer, size_output_layer, random_func, alpha=None, gamma=None, epsilon=None,
                 learning_rate=None, epsilon_min=None, epsilon_decay=None):
        self.random = random_func
        self._setHyperParameters(alpha, gamma, epsilon, learning_rate, epsilon_min, epsilon_decay)
        self._buildModel(size_input_layer, size_output_layer)
        self.memory = deque(maxlen=2000)


    def _buildModel(self, input_size, output_size):
        
       
        input_layer = layers.Input(shape=(input_size,))
        first_hidden_layer = layers.Dense(500, activation="relu")(input_layer)
        second_hidden_layer = layers.Dense(250, activation="relu")(first_hidden_layer)
        output_layer = layers.Dense(output_size, activation="linear")(second_hidden_layer)
        self.model = models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=Adam(lr=self._learning_rate), loss="mse")

    def _setHyperParameters(self, alpha, gamma, epsilon, learning_rate,epsilon_min, epsilon_decay):
        if learning_rate is not None:
            self._learning_rate = learning_rate
       
        super()._setHyperParameters(alpha, gamma, epsilon,epsilon_min, epsilon_decay)

    def _remember(self, current_state, action, reward, next_state, done):
        self.memory.append((current_state, action, reward, next_state, done))

    def _toInputarray(self, state):
        return np.array(state).reshape((1, -1))

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
        current_state = list(env.decode(env.s))
        action = self._explorationOrExploitation(current_state)

        next_state_id, reward, done, info = env.step(action)

        next_state = list (env.decode(next_state_id))

        self._remember(current_state, action, reward, next_state, done)

        # if len(self.memory) > self._batch_size:
        #     self._replay()

        if done:
            # env.render()
            # print("state",current_state)
            print("current Epsilon",self.epsilon)
            self._replay()
            self._updateEpsion()
            

        return action

    # https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def _replay(self):
        mini_batch = random.sample(self.memory, self._batch_size)
        array_states, array_nodes = [], []

        for current_state, action, reward, next_state, done in mini_batch:
         
            if not done:  
             pred= self._predictModel(next_state)
            #  print("pred",pred)
            #  print("argMax:",np.argmax(pred))  
             new_value = reward + self.gamma * np.amax(self._predictModel(next_state))
            else:
             new_value = 200
             pred = self._predictModel(current_state)
             
            
            nodes = self._predictModel(current_state)
            nodes[0][action] = new_value

            array_states.append(self._toInputarray(current_state)[0])
            array_nodes.append(nodes[0])

        
        history = self.model.fit(np.array(array_states), np.array(array_nodes), epochs=1, verbose=0).history
        loss = history["loss"][0]

        return loss

        

    def load(self, file):
        models.load_model(file)

    def save(self, file):
        self.model.save(file)
