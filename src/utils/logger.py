import numbers
import numpy as np 
from datetime import datetime
from tabulate import tabulate

class LearningLogger:
    def __init__(self,loss_keys=[],success_threshold_lookback=100):

        self.loss_keys=loss_keys
        self.success_threshold_lookback=success_threshold_lookback
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

    def episode_log_full(
        self,
        reward,
        running_reward,
        worker=False,
        success_strict=False
    ):
        data = []

        data.append(['Episodes',self.episodes])
        data.append(['Avg episode duration',str(round(np.mean(self.episode_durations),5))+' s'])
        data.append(['Learning steps',self.learning_steps])

        if type(worker) != bool:
            data.append(['Worker',worker])
        
        if success_strict != False:
            last_episodes = np.array(self.rewards[-self.success_threshold_lookback:])
            episodes_with_loss = len(last_episodes[last_episodes <=0])
            data.append(['Episodes on loss', '{}/{}'.format(episodes_with_loss,self.success_threshold_lookback)])
            #success_strict_str = "Total episodes ending on loss: {}/{} *".format(episodes_with_loss,self.success_threshold_lookback)
        
        data.append(['',''])
        data.append(['Last reward',round(reward,3)])
        data.append(['Avg reward',round(np.mean(self.rewards[-self.success_threshold_lookback:]),3)])
        data.append(['Running reward',round(np.mean(running_reward),3)])
        data.append(['',''])
        for loss in self.loss_keys:
            #data.append(['Last {} loss'.format(loss),round(self.losses[loss][-1],5)])
            data.append(['Avg {} loss'.format(loss),round(np.mean(self.losses[loss][-self.success_threshold_lookback:]),5)])

        print (tabulate(data, headers=["", ""]))
        print('')

    def episode_log_simple(self,
        reward,
        worker=False,
        success_strict=False
        ):
        
        worker_str=""
        success_strict_str=""

        rewards = np.array(self.rewards[-self.success_threshold_lookback:])
    
        episode_str = "Episode * {} * ".format(self.episodes)
        moving_avg_str = "Moving Avg Reward is ==> {:.5f} * ".format(np.mean(rewards))
        last_reward_str = "Last Reward was ==> {:.5f} ".format(reward)

        if success_strict != False:
            if len(rewards) > self.success_threshold_lookback:
                episodes_with_loss = len(rewards[rewards <=0])
                success_strict_str = "* Total episodes ending on loss: {}/{} ".format(episodes_with_loss,self.success_threshold_lookback)

        if type(worker) != bool:
            worker_str ="Worker #{} * ".format(worker)

        log_str = episode_str+worker_str+moving_avg_str+last_reward_str+success_strict_str

        print(log_str)

    def episode_test_log(self,score, episode=False):
        if episode != False:
            print("Episode * {} * Score ==> {:.3f}".format(episode, score))
        else:
            print('Episode done * Score ==> {:.3f}'.format(score))

    def episode(
        self,
        log_every, 
        reward, 
        running_reward, 
        log_level = 1, 
        worker=False, 
        success_strict=False
    ):
        # step it
        self.step_episode(reward,running_reward)

        # Log it 
        if self.episodes % log_every == 0 and self.episodes > 1:
            if log_level == 2:
                self.episode_log_full(
                    reward, 
                    running_reward,
                    worker=worker,
                    success_strict=success_strict)

            elif log_level == 1:
                self.episode_log_simple(
                    reward,
                    worker=worker,
                    success_strict=success_strict
                )

    def start_worker(self, worker_id):
        print("Worker * {} * has started.".format(worker_id))

        