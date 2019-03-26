import gym
import sys
sys.path.append("../")
from models import DeepQLearning
from gymBridge import GymBridge
import numpy as np 

from keras import layers
from keras import models
from keras.optimizers import Adam




def toInputArray(state):
    return np.reshape(state,(1,-1))


def buildModel(input_size, output_size):
        LEARNING_RATE = 0.001

        n_nodes = 1000        
        input_layer = layers.Input(shape=(input_size,))


        first_hidden_layer = layers.Dense(n_nodes, activation="relu")(input_layer)
        # second_hidden_layer = layers.Dense(n_nodes, activation="relu")(first_hidden_layer)
        # third_hidden_layer = layers.Dense(n_nodes,activation="relu")(second_hidden_layer)

        output_layer = layers.Dense(output_size, activation="linear")(first_hidden_layer)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss="mse")

        return model

def changeReward(next_observation, reward, done, info):
        if done:
          return -1

        return 1


if __name__ == "__main__":
    env = gym.make("CartPole-v1").env
    model = buildModel(env.observation_space.shape[0],env.action_space.n)
   
    deepQlearning  = DeepQLearning(model,env.action_space.sample,gamma=0.99, size_memory=10,epsilon_decay=0.996,batch_size=3,epsilon_min=0.1)

    bridge = GymBridge(env,deepQlearning,toInputArray,changeReward)

    bridge.trainAgent(debug=True, numb_of_matches=3600, max_steps=1000)

    rewards , timesteps = bridge.trainAgent(numb_of_matches=100,debug=True)
    
    print("average reward",rewards.mean())