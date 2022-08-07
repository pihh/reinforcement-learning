import numbers
import numpy as np 
from datetime import datetime
from tabulate import tabulate

class LearningLogger:
    def __init__(self,loss_keys=[]):

        self.loss_keys=loss_keys
        self.reset()
       

    def __getitem__(self, arg):
        return str(arg)

    def time_diff_seconds(self):
        now = datetime.now()
        prev = self.episode_start
        difference = (now - prev).total_seconds()
        self.episode_durations.append(difference)
        self.episode_start =  datetime.now()
        return difference

    def reset(self):
        self.episodes = 0
        self.learning_steps = 0
        self.episode_durations = []
        self.running_rewards = []
        self.rewards = []
        self.episode_start = datetime.now()

        # set loss trackers
        self.losses = {}
        for key in self.loss_keys:
            self.losses[key] = []

    def step_learning_steps(self):
        self.learning_steps +=1

    def step_loss(self, losses):
        valid = True
        for key in losses.keys():
            if not isinstance(losses[key],numbers.Number):
                valid = False

        if valid:
            for key in losses.keys():
                self.losses[key].append(losses[key])
            self.step_learning_steps()
        
    def step_episode(self, reward, running_reward):
        self.episodes += 1
        self.rewards.append(reward)
        self.running_rewards.append(running_reward)
        self.time_diff_seconds()

    def episode_log_full(self,reward,running_reward,worker=False):
        data = []

        data.append(['Episodes',self.episodes])
        data.append(['Avg episode duration',str(round(np.mean(self.episode_durations),5))+' s'])
        data.append(['Learning steps',self.learning_steps])
        if worker != False or worker== 0:
            data.append(['Worker',worker])
        data.append(['',''])
        data.append(['Last reward',round(reward,3)])
        data.append(['Avg reward',round(np.mean(self.rewards[-50:]),3)])
        data.append(['Running reward',round(np.mean(running_reward),3)])
        data.append(['',''])
        for loss in self.loss_keys:
            #data.append(['Last {} loss'.format(loss),round(self.losses[loss][-1],5)])
            data.append(['Avg {} loss'.format(loss),round(np.mean(self.losses[loss][-50:]),5)])

        print (tabulate(data, headers=["", ""]))
        print('')

    def episode_log_simple(self, reward,worker=False):
        
        worker_str=""
        episode_str = "Episode * {} * ".format(self.episodes)
        moving_avg_str = "Moving Avg Reward is ==> {:.5f} * ".format(np.mean(self.rewards[-50:]))
        last_reward_str = "Last Reward was ==> {:.5f}".format(reward)
        if worker != False or worker== 0:
            worker_str ="Worker * {} * ".format(worker)

        log_str = episode_str+worker_str+moving_avg_str+last_reward_str
        print(log_str)

    def episode_test_log(self,score, episode=False):
        if episode != False:
            print("Episode * {} * Score ==> {:.3f}".format(episode, score))
        else:
            print('Episode done * Score ==> {:.3f}'.format(score))

    def episode(self,log_each_n_episodes, reward,running_reward, log_level = 1,worker=False):
        # step it
        self.step_episode(reward,running_reward)

        # Log it 
        if self.episodes % log_each_n_episodes == 0 and self.episodes > 1:
            if log_level == 2:
                self.episode_log_full(reward, running_reward,worker)
            elif log_level == 1:
                self.episode_log_simple(reward,worker)

    def start_worker(self, worker_id):
        print("Worker * {} * has started.".format(worker_id))

        