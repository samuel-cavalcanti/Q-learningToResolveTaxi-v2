import gym
from IPython.display import clear_output
from time import sleep
from Qlearning import Q_learning
import numpy as np


# SOUTH = 0
# NORTH = 1
# EAST = 2
# WEST = 3
# PICKUP = 4
# DROPOFF = 5

class bruteForce:

    def execute(self, current_state):
        return np.random.randint(0,6)
    def update(self,*args):
        return
    def save(self, file):
        pass
    def load(self,file):
        pass

def initEnv(state=-1):
    global env
  
    env = gym.make("Taxi-v2").env
  

    if state != -1:
        env.s = state

    env.render(mode='ansi')


def trainAgent(agent,debug=False,numb_of_matches = 10000):
    for i in range(numb_of_matches):
        match(agent,debug)

    agent.save("q-table")



def saveState(state, action, reward, dic):
    dic.append(
        {
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
    )


def printFrames(frames):
    if frames is None:
        return

    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")

        sleep(.1)


def match(agent,debug=False):
 epochs = 0
 penalties = 0
 if debug:
    frames = []

 current_state = env.reset()

 done = False

 while not done:

    current_action = agent.execute(current_state)

    current_state, reward, done, info = env.step(current_action)
    
    agent.update(current_state,current_action,reward)


    if reward == -10:
        penalties += 1

    if debug:
        saveState(current_state, current_action, reward, frames)
        
    epochs += 1

   

 if debug:
    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))   
    return frames


def seeAgent(agent):
    agent.load("q-table.npy")
    frames = match(agent,True)
    printFrames(frames)



if __name__ == "__main__":
    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
    example_state = 328
    initEnv(example_state)
   # agent = bruteForce()
    agent = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample)


    #trainAgent(agent,True)
    seeAgent(agent)
    #random_agent_frames = bruteForce(True)


    # printFrames(random_agent_frames)

    pass


