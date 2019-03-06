import numpy as np

class Q_learning:
    alpha = 0.1
    gamma = 0.4
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.995

    def __init__(self, max_num_states, max_num_actions, random_func, alpha=None, gamma=None, epsilon=None, epsilon_min=None, epsilon_decay=None):
        self.q_table = np.zeros([max_num_states, max_num_actions])
        self.random = random_func
        self._setHyperParameters(alpha, gamma, epsilon, epsilon_min, epsilon_decay)

    def _setHyperParameters(self, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon
        if epsilon_min is not None:
            self.epsilon_min = epsilon_min
        if epsilon_decay is not None:
            self.epsilon_decay = epsilon_decay

    def _explorationOrExploitation(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.random()
        else:
            action = np.argmax(self.q_table[state])

        return action
    
    def action(self,state):
        self._current_state = state
        self._current_action = self._explorationOrExploitation(state)
        return self._current_action

    def update(self,next_state,reward,done,info):

        old_value = self.q_table[self._current_state, self._current_action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self._current_state, self._current_action] = new_value        

        if done:
            self._updateEpsion()

    def _updateEpsion(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
     
    def load(self, file):
        self.q_table = np.load(file)
        self.epsilon = 0.01

    def save(self, file):
        np.save(file, self.q_table)