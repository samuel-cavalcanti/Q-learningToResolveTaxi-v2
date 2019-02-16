import numpy as np


class Q_learning:
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    
    def __init__(self, max_num_states, max_num_actions,random_func,alpha=None,gamma=None,epsilon=None):
        self.q_table = np.zeros([max_num_states,max_num_actions])
        self.random = random_func

        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon

        pass


    def execute(self,current_state):
        self.current_state = current_state

        if np.random.uniform(0,1) < self.epsilon:
            action = self.random()
        else:
            action = np.argmax(self.q_table[current_state])

        return action


    def update(self,next_state,action,reward):
        old_value = self.q_table[self.current_state,action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.alpha )* old_value + self.alpha * (reward + self.gamma * next_max)

        self.q_table[self.current_state,action] = new_value

    def load(self,file):
        self.q_table = np.load(file)


    def save(self,file):
        np.save(file,self.q_table)


