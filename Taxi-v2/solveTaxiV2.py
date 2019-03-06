import gym
import sys
sys.path.append("../")
from gymBridge import GymBridge
from models import Q_learning



def toInputArray(state):
        return state

    

if __name__ == "__main__":
    env =  env = gym.make("Taxi-v2").env
    qlearning = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample)
    bridge = GymBridge(env,qlearning,toInputArray)
    bridge.trainAgent(debug=True)




    

