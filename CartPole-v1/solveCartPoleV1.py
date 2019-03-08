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
                
        input_layer = layers.Input(shape=(input_size,))

        first_hidden_layer = layers.Dense(48, activation="relu")(input_layer)
        second_hidden_layer = layers.Dense(48, activation="relu")(first_hidden_layer)

        output_layer = layers.Dense(output_size, activation="linear")(second_hidden_layer)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss="mse")

        return model

def changeReward(next_observation, reward, done, info):
        if done:
          return - reward

        return reward


if __name__ == "__main__":
    env = gym.make("CartPole-v1").env
    model = buildModel(env.observation_space.shape[0],env.action_space.n)
   
    deepQlearning  = DeepQLearning(model,env.action_space.sample,gamma=0.99, size_memory=None,epsilon_decay=0.996)

#     deepQlearning.load("model")


    bridge = GymBridge(env,deepQlearning,toInputArray,changeReward,limit_reward=1e3)

    bridge.trainAgent(debug=True, numb_of_matches=1200)

    rewards , timesteps = bridge.testAgent(numb_of_matches=100)
    
    print("average reward",rewards.mean())