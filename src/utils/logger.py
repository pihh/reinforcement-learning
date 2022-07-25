from datetime import datetime
from tabulate import tabulate
import numpy as np 

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
        self.episode = 0
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
        
    def step_episode(self, reward, running_reward, losses):
        self.episode += 1
        self.rewards.append(reward)
        self.running_rewards.append(running_reward)
        self.time_diff_seconds()

        for key in losses.keys():
            self.losses[key].append(losses[key])

    def log_episode(self, reward,running_reward,losses={}):
        # Episode
        # Avg episode duration
        # Learning steps 
        # Runinning reward
        # Avg reward 
        # Last reward
        # Running losses
        # Avg Losses
        # Last losses
        data = [[1, 'Liquid', 24, 12],
            [2, 'Virtus.pro', 19, 14],
            [3, 'PSG.LGD', 15, 19],
            [4,'Team Secret', 10, 20]]
        print (tabulate(data, headers=["Pos", "Team", "Win", "Lose"]))
        