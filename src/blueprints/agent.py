
import numpy as np


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input, Dense, Concatenate

from src.agents.agent import Agent
from src.utils.logger import LearningLogger
from utils.networks import MultiLayerPerceptron

class ActorCriticAgent(Agent):
    def __init__(self, 
                environment,
                fully_connected_layer_policy="mlp"
                
        ):
        
        super(ActorCriticAgent, self).__init__(environment)
        
        # Args
        self.fully_connected_layer_policy=fully_connected_layer_policy
        #type(tf.keras.optimizers.Adam(learning_rate=0.01)).__name__

        self.__init_networks()
        self.__init_buffers()
        
    def __init_buffers(self):
        #self.buffer = Buffer(['action_log_probs','critic_values','rewards'])
        pass

    def __init_networks(self):
        self.fully_connected_layer = MultiLayerPerceptron(policy=self.fully_connected_layer_policy)        
        if self.action_space_mode == "discrete":
            pass
        elif self.action_space_mode == "continuous":
            pass

        #self.model = keras.Model(inputs=inputs, outputs=[action, critic])

    def choose_action(self, state, deterministic=True):
        action = None 
        action_log_prob = None 
        critic_value = None
    
        if self.action_space_mode == "discrete":
            # DISCRETE SAMPLING
            if deterministic:
                pass
            else:
                pass

        elif self.action_space_mode == "continuous":
            # CONTINUOUS SAMPLING
            if deterministic:
                pass
            else:
                pass
        
        return action, action_log_prob , critic_value
    
    def test(self, episodes=10, render=True):

        for episode in range(episodes):
            # Reset and get state
            state = self.on_test_episode_start()

            done = False
            score = 0

            while not done:
                if render:
                    self.env.render()
                
                # Sample action, probs and critic
                action = self.choose_action(state)[0]

                # Step
                state,reward,done, _ = self.env.step(action)

                # Get next state
                score += reward

            # stops rendering and logs results
            self.on_test_episode_end(episode,score,render)
            

    
    def learn(self, timesteps=-1, plot_results=True, reset=False, success_threshold=False, log_level=1, log_each_n_episodes=50):
        
        success_threshold = self.on_learn_start(timesteps,success_threshold,reset)
 
        #self.buffer.reset()
        score = 0
        timestep = 0
        episode = 0
        
        while self.learning_condition(timesteps,timestep):  # Run until solved
            state = self.env.reset()
            score = 0
            done = False
            
            while not done:
                # Predict action probabilities and estimated future rewards from environment state
                action, action_log_prob, critic_value = self.choose_action(state, deterministic=False)

                # Store stuff in buffers
                    
                # Apply the sampled action in our environment
                state, reward, done, _ = self.env.step(action)

                # Update counters
                score += reward
                timestep+=1

            # End of episode    
            # Updates rewards, checks if success condition is met, logs results
            if self.on_learn_episode_end(score,log_each_n_episodes,log_level,success_threshold):
                break

        self.on_learn_end(plot_results)
