import numpy as np

class RunningReward:
    def __init__(
            self, 
            new_reward_weight=0.05, 
            old_reward_weight=0.95,
            success_threshold_lookback=50):
        """
        Tracks the moving reward of the agent on given episode
        
        Args:
        -------
        new_reward_weight: Weight given to new rewards ( the lower, the slower the moving reward evolves)
        old_reward_weight: Weight given to previous rewards
        success_threshold_lookback: How long to wait until we start using moving averages

        Methods
        -------
        reset():
            Restarts the counters(reward,episodes) and restores the historical trackers.

        step(reward):
            * Steps up one episode ( tracks how many eps has runned )
            * Updates running reward
            * Updates history ( running_reward and reward )
            * Updates moving average if at least n "success_threshold_lookback" steps have been made
        """
        assert new_reward_weight + old_reward_weight == 1 , "The sum of the old and new weights must be 1"

        self.new_reward_weight= new_reward_weight
        self.old_reward_weight= old_reward_weight
        self.success_threshold_lookback = success_threshold_lookback

        self.reset()

    def reset(self):
        self.reward = 0
        self.episodes = 0
        self.moving_average = -np.inf
        self.reward_history = []
        self.running_reward_history = []

    def step(self,reward):
        self.episodes +=1
        self.reward = self.new_reward_weight * reward + self.old_reward_weight * self.reward
        
        self.reward_history.append(reward)
        self.running_reward_history.append(self.reward)

        if len(self.reward_history) >= self.success_threshold_lookback:
            # np.mean(history[-100:])
            self.moving_average = np.mean(self.reward_history[-self.success_threshold_lookback:])