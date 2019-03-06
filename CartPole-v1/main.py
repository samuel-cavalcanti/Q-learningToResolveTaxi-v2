import gym

from DeepQLearning import DeepQLearning ,np


def initEnv():
    global env
    env = gym.make("CartPole-v1").env

    return  env.observation_space.shape[0], env.action_space.n 

def match(agent,debug):
    env.reset()

    for time in range(500):
      pass





def trainAgent(agent,number_episodes=1000,debug=False):
    for i in range(number_episodes):
        match(agent,debug)
    
    pass

if __name__ == "__main__":
    input_size, output_size = initEnv()

    while True:
        env.render()
        
    
    agent = DeepQLearning(input_size,output_size,np.random.rand)

    
    pass