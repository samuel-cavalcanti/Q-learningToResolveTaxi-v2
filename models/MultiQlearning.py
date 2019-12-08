from .Qlearning import np, Q_learning


class MultiQlearning(Q_learning):
    __accumulate_reward = 0
    __n_iterations = 0
    _choose = 0

    def __init__(self, size_population: int, max_num_states: int, max_num_actions: int, random_func,
                 alpha=0.2, gamma=0.9, epsilon=1, epsilon_min=0.01,
                 epsilon_decay=0.9995, n_resets=0):

        self.random = random_func

        self._current_state = None

        self._current_action = None

        self._population = self._generatePopulation(size_population, max_num_states, max_num_actions)

        super()._setHyperParameters(alpha, gamma, epsilon, epsilon_min, epsilon_decay, n_resets)

    def _generatePopulation(self, size_population, max_num_states, max_num_actions):
        return [self._generateQTable(max_num_states, max_num_actions) for i in range(size_population)]

    def _explorationOrExploitation(self, state):
        self._choose = np.random.randint(0, len(self._population))

        if np.random.uniform(0, 1) < self._epsilon:
            action = self.random()
        else:
            action = np.argmax(self._population[self._choose][state])

        return action

    def action(self, state):
        self._current_state = state
        self._current_action = self._explorationOrExploitation(state)

        return self._current_action

    def update(self, next_state, reward, done, info):

        self.__accumulate_reward += reward

        self._updateQTable(self._choose, next_state, reward)

        if done:
            self._updateEpsilon()
            self.__n_iterations += 1

    def _updateQTable(self, index_table, next_state, reward):
        old_value = self._population[index_table][self._current_state, self._current_action]

        next_max = self._nextMax(index_table, next_state)
        n = 1 / (len(self._population) - 1)

        new_value = old_value + self._alpha * (reward + self._gamma * n * next_max)

        self._population[index_table][self._current_state, self._current_action] = new_value

    def _resetEvaluateParameters(self):
        self.__accumulate_reward = 0
        self.__n_iterations = 0

    def _nextMax(self, index, next_state):
        s = 0
        a = np.argmax(self._population[index][next_state])

        for q_table in self._population:
            if q_table is not self._population[index]:
                s += q_table[next_state][a] - self._population[index][self._current_state][self._current_action]

        return s
