import gym

import sys
sys.path.append("../")
from models import Q_learning, EvolutionaryQlearning, MultiQlearning
from gymBridge import GymBridge
import numpy as np

from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


def toInputArray(state):
    return state

def generateInterval():
    first = list()
    for i in range(env.observation_space.n):
        second = list()
        for j in range(env.action_space.n):
            second.append([0, 1])
        first.append(second)

    return np.array(first)


def testEvolutionaryQlearning():
    size_population = 4

    interval = generateInterval()

    model = EvolutionaryQlearning(size_population, interval, float, env.action_space.sample, time_to_evolve=100,
                                  epsilon=1)

    bridge = GymBridge(env, model, toInputArray)

    bridge.trainAgent(numb_of_matches=1e4, max_steps=250, terminal_debug=True, show_plot=True)

    model._epsilon = 0

    time_steps, rewards = bridge.trainAgent(numb_of_matches=1e3)

    print("mean reward: {}".format(rewards.mean()))
    print("mean steps: {}".format(time_steps.mean()))


def playFrozen():
    env.reset()

    while True:
        env.render()
        mov = int(input("escolha uma ação\n"))
        if mov == 40:
            break
        next_observation, reward, done, info = env.step(mov)
        print("next_observation: {}, reward : {}, done: {}, info: {}".format(next_observation, reward, done, info))
        if done:
            env.reset()


def printPolicy(table):
    actions = {0: "Left ",
               1: "Down ",
               2: "Right",
               3: "Up   "}

    best_actions = [np.argmax(state) for state in table]

    for i in range(len(best_actions)):
        if i == 5 or i == 7 or i == 11 or i == 12:
            print("Hole ", end=" ")
        elif i == 15:
            print("Goal ")
        else:
            print(actions[best_actions[i]], end=" ")

        if (i + 1) % 4 == 0:
            print("\n")


def printMultiPolicy(tables):
    for table in tables:
        printPolicy(table)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    # playFrozen()

    # 0->left
    # 1-> down0
    # 2-> right
    # 3-> up

  
    model = MultiQlearning(4, env.observation_space.n, env.action_space.n, env.action_space.sample)

    bridge = GymBridge(env, model, toInputArray)

    bridge.trainAgent(numb_of_matches=1e5, max_steps=250, terminal_debug=True, show_plot=False)
    model._epsilon = 0
    time_steps, rewards = bridge.trainAgent(numb_of_matches=1e3, show_plot=True)
    
    print("mean reward: {}".format(rewards.mean()))
    print("mean steps: {}".format(time_steps.mean()))



