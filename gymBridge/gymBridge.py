from matplotlib import pyplot as plot
import numpy as np 

class GymBridge:
    
    def __init__(self, gym_env,model,toInputArray,change_reward=None,limit_reward = float("inf")):
        self._env = gym_env
        self._model = model
        self._toInputArray = toInputArray
        self._changeReward = change_reward
        self._limit_reward = limit_reward

    
    def oneStep(self,test_agent):

        action = self._model.action(self._current_observation)

        next_observation, reward, done, info  = self._env.step(action)

        self._current_observation = self._toInputArray(next_observation)

        if self._changeReward is not None:
            reward = self._changeReward(next_observation, reward, done, info)

        if not test_agent:
            self._model.update(self._current_observation,reward,done,info)

        

        return done , reward , info

    def match(self,max_steps= float("inf"),test_agent = False ):
        total_reward = 0 
        steps = 0
        done = False
        self._current_observation = self._toInputArray( self._env.reset() ) 

        while steps <  max_steps and not done:
            done, reward, info = self.oneStep(test_agent)
            total_reward += reward
            steps += 1
            if total_reward > self._limit_reward:
                return steps , total_reward , done

        return steps , total_reward , done


    def trainAgent(self,debug=False, numb_of_matches=10000,max_steps = float("inf"), model_file = "model" ):
        if debug:
          time_steps = []
          reward_list = []
          avarage_reward = 0

        for i in range(int(numb_of_matches)):
          steps , reward, done = self.match(max_steps)

          if debug:
                time_steps.append(steps)
                reward_list.append(reward)

                if len(reward_list) % 100 == 0:
                    avarage_reward = np.mean(reward_list[i-100:i])
                self._terminalDebug(i,numb_of_matches,steps,reward,avarage_reward)
                
               

        if debug:
            self._showGraph(time_steps,reward_list)

        if type(model_file) is str:
            self._model.save(model_file)
        

    def _terminalDebug(self,current_episode,max_episode,steps,reward,avarage_reward ):
         print("episode: {}/{}".format(current_episode,max_episode))
         print("Timesteps taken: {}".format(steps))
         print("reward: {}".format(reward))
         print("avarage reward {}".format(avarage_reward))


    def _showGraph(self,time_steps,reward_list):
        time_steps = np.array(time_steps)
        reward_list= np.array(reward_list)

    
       

        plot.figure("Timesteps")
        plot.plot(time_steps)
        plot.ylabel("movimentos")
        plot.xlabel("partida")

        plot.figure("reward")
        plot.plot(reward_list)
        plot.xlabel("partida")
        plot.ylabel("recompesa")


        # plot.figure("average reward ")
        # plot.plot()
        # plot.xlabel("partida")
        # plot.ylabel("média de recompesa")


        plot.show()
       
    def testAgent(self,debug=False,numb_of_matches =200,max_steps = float("inf")):
        time_steps = []
        reward_list = []

        for i in range(numb_of_matches):
            steps , reward, done = self.match(max_steps)

            if debug :
                print("episode: {}/{}".format(i,numb_of_matches))
                print("Timesteps taken: {}".format(steps))
                print("reward: {}".format(reward))
                print("finish match: {}".format(done))

            time_steps.append(steps)
            reward_list.append(reward)

        self._showGraph(time_steps,reward_list)

        return  np.array(reward_list) , np.array(time_steps)