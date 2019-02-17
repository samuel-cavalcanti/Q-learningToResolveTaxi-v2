import gym
from IPython.display import clear_output
from time import sleep
from Qlearning import Q_learning
from DeepQLearning import DeepQLearning
import numpy as np
from matplotlib import pyplot as plot


# SOUTH = 0
# NORTH = 1
# EAST = 2
# WEST = 3
# PICKUP = 4
# DROPOFF = 5

class bruteForce:

    def execute(self, env):
        action = np.random.randint(0, 6)
        env.step(action)
        return action

    def _update(self, *args):
        return

    def save(self, file):
        pass

    def load(self, file):
        pass


def initEnv(state=-1):
    global env

    env = gym.make("Taxi-v2").env

    if state != -1:
        env.s = state

    env.render(mode='ansi')


def trainAgent(agent, debug=False, numb_of_matches=10000):
    # try:
    #     agent.load("q-table.npy")
    # except:
    #     pass

    global time_steps, penalties_list
    time_steps = []
    penalties_list = []


    for i in range(numb_of_matches):
        match(agent, debug)



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

        sleep(.5)


def match(agent, debug=False):
    epochs = 0
    penalties = 0
    if debug:
        frames = []
        plot.figure("Timesteps")
        plot.plot(time_steps)
        plot.figure("Penalties")
        plot.plot(penalties_list)
        plot.show()
        plot.pause(0.00000000000001)
        plot.clf()

    env.reset()

    done = False

    while not done and epochs < 256 and penalties < 100:

        action = agent.execute(env)

        reward = env.P[env.s][action][0][2]
        done = env.P[env.s][action][0][3]

        if reward == -10:
            penalties += 1

        if debug:
            saveState(env.s, action, reward, frames)

        epochs += 1

    if debug:
        time_steps.append(epochs)
        penalties_list.append(penalties)

        print("Timesteps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))
        return frames


def seeAgent(agent):
    agent.load("q-table.npy")
    frames = match(agent, True)
    printFrames(frames)


if __name__ == "__main__":

    # impedir o pyplot de congelar meu processo atual
    plot.ion()

    # https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
    example_state = 328
    initEnv(example_state)
    size_of_possible_actions = int(len(env.P[env.s]) * (len(env.P[env.s][0][0])))

    # agent = bruteForce()
    agent = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample)
    #agent = DeepQLearning(size_of_possible_actions, env.action_space.n, env.action_space.sample, epsilon=1, gamma=0.95)

    # print(env)
    trainAgent(agent, True)
    # seeAgent(agent)
    # random_agent_frames = bruteForce(True)

    # printFrames(random_agent_frames)

    pass
