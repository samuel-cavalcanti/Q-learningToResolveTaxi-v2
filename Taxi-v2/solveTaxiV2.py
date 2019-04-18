import gym

import signal
import sys

sys.path.append("../")

from gymBridge import GymBridge
from models import Q_learning, EvolutionaryQlearning , MultiQlearning

import numpy as np


def toInputArray(state):
    return state

def generateInterval():
    first = list()
    for i in range(env.observation_space.n):
        second = list()
        for j in range(env.action_space.n):
            second.append([-10, 10])
        first.append(second)

    return np.array(first)


def testEvolutionaryQlearning():
    size_population = 4

    interval = generateInterval()

    model = EvolutionaryQlearning(size_population, interval, float, env.action_space.sample, time_to_evolve=100)

    # model = MultiQlearning(size_population,env.observation_space.n,env.action_space.n,env.action_space.sample)

    bridge = GymBridge(env, model, toInputArray)

    bridge.trainAgent(numb_of_matches=1e4, max_steps=250, terminal_debug=True, show_plot=True)

    model._epsilon = 0

    time_steps, rewards = bridge.trainAgent(numb_of_matches=1e3)

    print("mean reward: {}".format(rewards.mean()))
    print("mean steps: {}".format(time_steps.mean()))


if __name__ == "__main__":
    env = env = gym.make("Taxi-v2").env

    
    model = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample, alpha=0.2, gamma=0.9)
    bridge = GymBridge(env, model, toInputArray)
    bridge.trainAgent(numb_of_matches=1e4, max_steps=250)
    model._epsilon = 0
    time_steps, rewards = bridge.trainAgent(numb_of_matches=1e3)
    
    print("mean reward: {}".format(rewards.mean()))
    print("mean steps: {}".format(time_steps.mean()))