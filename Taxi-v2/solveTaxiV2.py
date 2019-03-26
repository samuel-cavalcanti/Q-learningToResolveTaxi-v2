import gym
import sys
import signal

sys.path.append("../")
from gymBridge import GymBridge
from models import Q_learning
from geneticAlgorithm import GeneticAlgorithm



def toInputArray(state):
    return state


def changeReward(next_observation, reward, done, info):
    return reward


def fitnessFunction(chromosome):
   
    model = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample,
                       alpha=chromosome[0], gamma=chromosome[1], epsilon_decay=chromosome[2], epsilon_min=chromosome[3])

    bridge = GymBridge(env, model, toInputArray)


    time_steps, rewards = bridge.trainAgent(numb_of_matches=1e5, max_steps=250, showPlot=True)
    


    return - rewards.mean()

def ctrl_c(signal,frame):
    
    ga.print()

    shutdown = input("deseja cancelar o algoritmo, y/n ? \n")
    
    save_ga = input("\n deseja salvar a população, y/n ? ")


    if save_ga == "y":
        ga.populationToCSV("individuals.csv")    


    if shutdown == "y":
        exit(0)

    print("OK, voltando para execução do algoritmo")

if __name__ == "__main__":
    env = env = gym.make("Taxi-v2").env
    
    signal.signal(signal.SIGINT,ctrl_c)
   


    #     model = Q_learning(env.observation_space.n, env.action_space.n, env.action_space.sample,alpha=0.2,gamma=0.8589,epsilon_decay=0.996,I=0.01,n_resets=10)
    #     model.load("model.npy")

    interval = [[0, 1], [0, 1], [1.0, 0.7], [0.5, 0]]
    size_population = 5
   
    ga = GeneticAlgorithm(size_population, interval, float, fitnessFunction)

   

    solution = ga.execute()

    while solution.score > -9:
        solution = ga.execute()
        print(solution.score)

    print(solution.chromosome)
    solution.save("best_solution")
   

#     bridge = GymBridge(env,model,toInputArray,change_reward=changeReward)
#     bridge.trainAgent(debug=True,numb_of_matches=4e4)
#     time_steps,rewards = bridge.trainAgent(debug=True,numb_of_matches=100)


#     print("average reward",rewards.mean())
