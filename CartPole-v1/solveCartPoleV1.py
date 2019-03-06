import gym
import sys
sys.path.append("../")
from models import DeepQLearning
from gymBridge import GymBridge
import numpy as np 

from keras import layers
from keras import models
from keras.optimizers import Adam


LEARNING_RATE = 0.001

def toInputArray(state):
    return np.reshape(state,(1,-1))


def buildModel(input_size, output_size):
                
        input_layer = layers.Input(shape=(input_size,))

        first_hidden_layer = layers.Dense(24, activation="relu")(input_layer)
        second_hidden_layer = layers.Dense(48, activation="relu")(first_hidden_layer)

        output_layer = layers.Dense(output_size, activation="linear")(second_hidden_layer)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss="mse")

        return model



if __name__ == "__main__":
    env = gym.make("CartPole-v1").env
    model = buildModel(env.observation_space.shape[0],env.action_space.n)

    deepQlearning  = DeepQLearning(model,env.action_space.sample)

    bridge = GymBridge(env,deepQlearning,toInputArray)

    bridge.trainAgent(debug=True, numb_of_matches=5000)
