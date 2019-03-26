from matplotlib import pyplot as plot
from matplotlib.animation import FuncAnimation
import numpy as np 


class GymBridge:
    

    def __init__(self, gym_env,model,toInputArray,change_reward=None,limit_reward = float("inf")):
        self._env = gym_env
        self._model = model
        self._toInputArray = toInputArray
        self._changeReward = change_reward
        self._limit_reward = limit_reward
        #futura classe

        self.__steps = list()
        self.__rewards = list()

    
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


    def trainAgent(self,showPlot=False , terminalDebug= False, numb_of_matches=10000,max_steps = float("inf"), model_file = None ):
        if terminalDebug:
          avarage_reward = 0
          sample_size = 100



        time_steps = []
        reward_list = []

        for i in range(int(numb_of_matches)):
          steps , reward, done = self.match(max_steps)
          self.__steps.append(steps)
          self.__rewards.append(reward)
          

         

          time_steps.append(steps)
          reward_list.append(reward)

          if terminalDebug:
                if len(reward_list) % sample_size == 0:
                    avarage_reward = np.mean(reward_list[i+1-sample_size:i])
                self._terminalDebug(i,numb_of_matches,steps,reward,avarage_reward)
                
               

        # if showPlot:
        #     plot.pause(0.05)
        #     self._showGraph(time_steps,reward_list,show=True)

           

        if type(model_file) is str:
            self._model.save(model_file)
        
        return np.array(time_steps) , np.array(reward_list)

    def _terminalDebug(self,current_episode,max_episode,steps,reward,avarage_reward ):
         print("episode: {}/{}".format(current_episode,max_episode))
         print("Timesteps taken: {}".format(steps))
         print("reward: {}".format(reward))
         print("avarage reward {}".format(avarage_reward))
         if hasattr(self._model,"getEpsilon"):
                print("curret epsilon: {}".format(self._model.getEpsilon()))

    def _plotGraph(self,array,plot_name,x_label,y_label,show,save):

        
        plot.figure(plot_name)
        plot.plot(array)
        plot.ylabel(x_label)
        plot.xlabel(y_label)

        if save:
            plot.savefig(plot_name,bbox_inches="tight")

        if show:
           
            plot.draw()
           
         

    def _showGraph(self,time_steps,reward_list,show=False,save=False):
        plot.clf()
        self._plotGraph(time_steps,"Time steps","steps","match",False,save)
        self._plotGraph(reward_list,"reward","reward","match",show,save)




# futura classe
    def realTimePlot(self):
      
        

        #futura classe
        self.__reawrd_plot = plot.figure("Reward")
        self.__time_steps_plot = plot.figure("Time Steps")
       
        self.__time_steps_space, = plot.plot([], [], "-")
        self.__reward_space, = plot.plot([], [], "-")


        self.animation = FuncAnimation(self.__reawrd_plot,self._updateReward,blit=True)
        # self.animation_2 = FuncAnimation(self.__time_steps_plot,self._updateTimeSteps,interval=200)
        plot.show()
        
        
       
       

     


    def closePlot(self):
        plot.close(self.__reawrd_plot)
        plot.close(self.__time_steps_plot)
        print("finish")
        plot.ioff()



    def _updateReward(self,frame):
       
        self.__reawrd_plot.gca().relim()
        self.__reawrd_plot.gca().autoscale_view()

      

        print("__rewards",frame)
        axis_x = [ i for i in range(len(self.__rewards)) ]

        print("re",axis_x)
      

        self.__reward_space.set_data(axis_x, self.__rewards)

           
    
        return self.__reward_space,

    def _updateTimeSteps(self,frame):
        print("steps")

        axis_x = [ i for i in range(len(self.__steps)) ]
        self.__time_steps_space.set_data(axis_x,self.__steps)

        self.__time_steps_plot.gca().relim()
        self.__time_steps_plot.gca().autoscale_view()

       

        return self.__time_steps_space,