import numpy as np


class Q_learning:
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    def __init__(self, max_num_states, max_num_actions, random_func, alpha=None, gamma=None, epsilon=None):
        self.q_table = np.zeros([max_num_states, max_num_actions])
        self.random = random_func
        self._setHyperParameters(alpha, gamma, epsilon)

    def _setHyperParameters(self, alpha, gamma, epsilon):
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon

    def _explorationOrExploitation(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.random()
        else:
            action = np.argmax(self.q_table[state])

        # action = np.argmax(self.q_table[state])

        return action

    def execute(self, env):
        self.current_state = env.s
        current_state = env.s

        action = self._explorationOrExploitation(current_state)

        next_state, reward, done, info = env.step(action)

        self._update(current_state, next_state, action, reward)

        return action

    def _update(self, current_state, next_state, action, reward):
        old_value = self.q_table[current_state, action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[current_state, action] = new_value

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        # print("epsilon:",self.epsilon)

    def load(self, file):
        self.q_table = np.load(file)

    def save(self, file):
        np.save(file, self.q_table)
