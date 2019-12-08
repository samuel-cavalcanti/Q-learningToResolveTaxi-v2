import pylab
import numpy as np
import time


class GymBridge:

    def __init__(self, gym_env, model, toInputArray, change_reward=None, limit_reward=float("inf")):
        self._env = gym_env
        self._model = model
        self._toInputArray = toInputArray
        self._changeReward = change_reward
        self._limit_reward = limit_reward
        self._current_observation = None
        self._current_action = None
        # futura classe

        self.__average_steps = list()
        self.__average_reward = list()

    def oneStep(self, test_agent):

        action = self._model.action(self._current_observation)
        if test_agent:
            self._env.render()

        next_observation, reward, done, info = self._env.step(action)

        if test_agent:
            self._env.render()
            # print("action: {}".format(action))
            # time.sleep(1.5)

        if self._changeReward is not None:
            reward = self._changeReward(self._current_observation, next_observation, reward, done, info)

        self._current_observation = self._toInputArray(next_observation)

        if not test_agent:
            self._model.update(self._current_observation, reward, done, info)

        return done, reward, info

    def match(self, max_steps=float("inf"), test_agent=False):
        total_reward = 0
        steps = 0
        done = False
        self._current_observation = self._toInputArray(self._env.reset())

        while steps < max_steps and not done:
            done, reward, info = self.oneStep(test_agent)
            total_reward += reward
            steps += 1
            if total_reward > self._limit_reward:
                return steps, total_reward, done

        return steps, total_reward, done

    def trainAgent(self, show_plot=False, terminal_debug=False, numb_of_matches=10000, max_steps=float("inf"),
                   model_file=None, test_agent=False):
        if terminal_debug:
            sample_size = 1

        time_steps = []
        reward_list = []

        for i in range(int(numb_of_matches)):
            steps, reward, done = self.match(max_steps, test_agent)

            time_steps.append(steps)
            reward_list.append(reward)

            if terminal_debug and len(reward_list) % sample_size == 0:
                head = i + 1 - 100
                average_reward = np.mean(reward_list[head:i])

                average_steps = np.mean(time_steps[head:i])
                self._terminalDebug(i, numb_of_matches, steps, reward, average_reward)
                self.__average_steps.append(average_steps)
                self.__average_reward.append(average_reward)
                # print("Time to Evolve: {}".format(self._model._time_to_evolve))

        if show_plot:
            self._showGraph(time_steps, reward_list, show=True)

        if type(model_file) is str:
            self._model.save(model_file)

        return np.array(time_steps), np.array(reward_list)

    def _terminalDebug(self, current_episode, max_episode, steps, reward, average_reward):
        print("episode: {}/{}".format(current_episode, max_episode))
        print("Timesteps taken: {}".format(steps))
        print("reward: {}".format(reward))
        print("avarage reward {}".format(average_reward))
        if hasattr(self._model, "_epsilon"):
            print("curret epsilon: {}".format(self._model.getEpsilon()))

    def _plotGraph(self, array, plot_name, x_label, y_label, show, save):

        pylab.figure(plot_name)
        pylab.plot(array)
        pylab.ylabel(x_label)
        pylab.xlabel(y_label)

        if save:
            pylab.savefig(plot_name, bbox_inches="tight")

        if show:
            pylab.show()

    def _showGraph(self, time_steps, reward_list, show=False, save=False):
        self._plotGraph(self.__average_steps, "Average Steps", "steps", "math", False, save)
        self._plotGraph(self.__average_reward, "Average reward", "reward", "match", False, save)
        self._plotGraph(time_steps, "Time steps", "steps", "match", False, save)
        self._plotGraph(reward_list, "Rreward", "reward", "match", show, save)
