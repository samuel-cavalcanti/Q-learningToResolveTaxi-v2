from matplotlib import pyplot as plot

class GymBridge:
    
    def __init__(self, gym_env,model,toInputArray):
        self._env = gym_env
        self._model = model
        self._toInputArray = toInputArray

    
    def oneStep(self):

        action = self._model.action(self._current_observation)

        next_observation, reward, done, info  = self._env.step(action)

        self._current_observation = self._toInputArray(next_observation)

        self._model.update(self._current_observation,reward,done,info)

        

        return done , reward , info

    def match(self,max_steps= float("inf") ):
        total_reward = 0 
        steps = 0
        done = False
        self._current_observation = self._toInputArray( self._env.reset() ) 

        while steps <  max_steps and not done:
            done, reward, info = self.oneStep()
            total_reward += reward
            steps += 1

        return steps , total_reward , done


    def trainAgent(self,debug=False, numb_of_matches=10000,max_steps = float("inf"), model_file = "model" ):
        if debug:
          time_steps = []
          reward_list = []

        for i in range(numb_of_matches):
          steps , reward, done = self.match(max_steps)
          if debug:
               print("episode: {}/{}".format(i,numb_of_matches))
               print("Timesteps taken: {}".format(steps))
               print("reward: {}".format(reward))
               print("finish match: {}".format(done))
               time_steps.append(steps)
               reward_list.append(reward)

        if debug:
            self._showGraph(time_steps,reward_list)

        if type(model_file) is str:
            self._model.save(model_file)
        

    def _showGraph(self,time_steps,reward_list):

        plot.figure("Timesteps")
        plot.plot(time_steps)
        plot.ylabel("movimentos")
        plot.xlabel("partida")

        plot.figure("reward")
        plot.plot(reward_list)
        plot.xlabel("partida")
        plot.ylabel("recompesa")

        plot.show()
       
        

