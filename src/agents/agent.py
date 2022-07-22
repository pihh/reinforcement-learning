import numpy as np
import matplotlib.pyplot as plt

from src.utils.running_reward import RunningReward
from src.utils.gym_environment import GymEnvironment

class Agent:
    def __init__(self,
        environment,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.00001,
    ):
        # Args
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.epsilon_ = epsilon
        self.epsilon_decay_ = epsilon_decay
        
        # Boot
        self.__init_environment(environment)
        self.__init_reward_tracker()
        

    def __init_environment(self,environment):
        env = GymEnvironment(environment)
        self.env = env.env
        self.n_actions = env.n_actions
        self.n_inputs = env.n_inputs
        self.actions = env.actions
        self.observation_shape = env.observation_shape
        self.action_space_mode = env.action_space_mode
        self.action_upper_bounds = env.action_upper_bounds
        self.action_lower_bounds = env.action_lower_bounds

    def __init_reward_tracker(self):
        self.running_reward = RunningReward()
        
    def validate_learn(self,timesteps, success_threshold, reset):
        if reset:
            self.__init_reward_tracker()
            self.epsilon = self.epsilon_
            if timesteps > -1:
                self.epsilon_decay = 2/timesteps
        
            
        assert hasattr(self.env,'success_threshold') or success_threshold and timesteps==-1 , "A success threshold is required for the environment to run indefinitely"

    # Loop condition
    def learning_condition(self,timesteps,timestep):
        if timesteps == -1:
            return True
        else: 
            return timesteps > timestep

    def did_finnish_learning(self,success_threshold,episode):
        # Break loop if average reward is greater than success threshold
        if self.running_reward.moving_average > success_threshold and episode > 10:
            print('Agent solved environment at the episode {}'.format(episode))
            return True
        return False

    # Tests
    def test(self, episodes=10, render=True):

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if render:
                    self.env.render()
                    
                action = self.choose_action(state)

                # Step
                state,reward,done, info = self.env.step(action)

                # Get next state
                score += reward
            
            if render:
                self.env.close()

            print("Test episode: {}, score: {:.2f}".format(episode,score))

    def decrement_epsilon(self):
        #self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > 0.01 else 0.01
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def plot_learning_results(self):
        plt.plot(self.running_reward.reward_history)

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
