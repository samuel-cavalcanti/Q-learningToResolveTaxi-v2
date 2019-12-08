import gym
import sys

sys.path.append("../")
from models import DeepQLearning
from gymBridge import GymBridge
import numpy as np

from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.metrics import mse


def toInputArray(current_state: np.ndarray, old_state=None, current_action=np.array([0, 0]),
                 old_action=np.array([0, 0])):
    if old_state is None:
        input_array = np.array(current_state.tolist() + current_action + current_state.tolist() + old_action).reshape(
            (2, -1))
        # print(input_array.shape)
        # print(input_array)
        # exit(1)
    else:
        input_array = np.array(old_state.tolist() + old_action + current_state.tolist() + current_action). \
            reshape((2, -1))

    return input_array


def buildModel(input_size, output_size):
    n_nodes = 100
    # input_layer = layers.Input(shape=(input_size,))
    #
    # # lstm = layers.LSTM(34, batch_input_shape=(1, input_size))
    #
    # first_hidden_layer = layers.Dense(n_nodes, activation="relu")(input_layer)
    # # lstm = layers.LSTM(32)(first_hidden_layer)
    # second_hidden_layer = layers.Dense(n_nodes, activation="relu")(first_hidden_layer)
    # third_hidden_layer = layers.Dense(n_nodes, activation="relu")(second_hidden_layer)
    #
    # output_layer = layers.Dense(output_size, activation="linear")(third_hidden_layer)
    #
    # model = models.Model(inputs=input_layer, outputs=output_layer)
    #
    # model = models.Sequential()
    # model.add(layers.LSTM(34, input_shape=(1, 4)))
    # model.add(layers.Dense(output_size, activation="linear"))

    model = models.Sequential()

    input_shape = (2, input_size + output_size)

    model.add(layers.SimpleRNN(n_nodes, input_shape=input_shape))
    model.add(layers.Dense(input_size + output_size))

    model.compile(optimizer=Adam(), loss="mse", metrics=[mse])

    return model


def changeReward(current_observation, next_observation, reward, done, info):
    # print("current_observation {}".format(current_observation))
    # print("next_observation {}".format(next_observation))
    # print("euclidian distance: {}".format(np.linalg.norm(current_observation - next_observation)))

    return reward  # / np.linalg.norm(current_observation - next_observation)
    # return np.sum(next_observation)


if __name__ == "__main__":
    env = gym.make("CartPole-v1").env
    model = buildModel(env.observation_space.shape[0], env.action_space.n)

    deepQlearning = DeepQLearning(model, env.action_space.sample, gamma=0.99, size_memory=50, epsilon_decay=0.996,
                                  batch_size=3, epsilon_min=0.1)
    # deepQlearning.load("lstm")

    bridge = GymBridge(env, deepQlearning, toInputArray, changeReward)

    bridge.trainAgent(terminal_debug=True, numb_of_matches=550)

    rewards, timesteps = bridge.trainAgent(numb_of_matches=100, terminal_debug=True, show_plot=True)

    print("average reward", rewards.mean())

    # deepQlearning.save("NARX")
