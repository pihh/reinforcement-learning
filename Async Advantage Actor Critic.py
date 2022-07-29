#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutup
shutup.please()


# In[2]:


import os
import numpy as np
import random
from datetime import datetime
from multiprocessing import cpu_count
from threading import Thread

from src.agents.agent import Agent
from src.utils.buffer import Buffer

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Model

from tensorflow.python.framework.ops import disable_eager_execution

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
disable_eager_execution()



import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.size = 0

    def reset(self):
        self.size = 0
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def remember(self, state, action_onehot, reward ):
        self.size +=1
        self.states.append(state)
        self.actions.append(action_onehot)
        self.rewards.append(reward)

#     def sample(self, batch_size=64):
#         max_mem = min(self.buffer_counter, self.buffer_size)

#         batch = np.random.choice(max_mem, batch_size)

#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.done_memory[batch]

#         return states, actions, rewards, states_, dones


# In[8]:


GLOBAL_EPISODE_NUM = 0

class A3CWorker(Thread):
    def __init__(self, 
                worker_id,
                env, 
                global_actor, 
                global_critic, 
                action_space_mode,
                observation_shape,
                policy,
                n_actions,
                actor_optimizer,
                critic_optimizer,
                std_bound,
                max_episodes = 10000):
        Thread.__init__(self)
        
        self.worker_id = worker_id
        self.env = env
        self.action_space_mode = action_space_mode
        self.observation_shape = observation_shape
        self.policy = policy
        self.n_actions = n_actions
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.std_bound = std_bound
        self.max_episodes = 10000

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.__init_networks()
        self.__init_buffers()

        self.actor.set_weights(self.global_actor.get_weights())
        self.critic.set_weights(self.global_critic.get_weights())

        self.actor._make_predict_function()
        self.critic._make_predict_function()

        # self.global_actor.summary()
        # self.actor.summary()

    def __init_buffers(self):
        self.buffer = ReplayBuffer()

    def __init_networks(self):
        X_input_actor = Input(shape=self.observation_shape) 
        X_actor = CommonLayer(X_input_actor,self.policy,rename=False)

        X_input_critic = Input(shape=self.observation_shape) 
        X_critic = CommonLayer(X_input_critic,self.policy,rename=False)
        
        action = Dense(self.n_actions, activation="softmax", kernel_initializer='he_uniform')(X_actor)
        value = Dense(1, kernel_initializer='he_uniform')(X_critic)
        
        if self.action_space_mode == "discrete":
            action = Dense(self.n_actions, activation="softmax", kernel_initializer='he_uniform')(X_actor)
            self.actor = Model(inputs = X_input_actor, outputs = action)
            self.actor.compile(loss='categorical_crossentropy', optimizer=self.actor_optimizer)
        else:
            mu = Dense(self.n_actions, activation="tanh", kernel_initializer='he_uniform')(X_actor)
            mu = Lambda(lambda x: x * self.action_bound)(mu)
            sigma = Dense(self.n_actions, activation="softplus", kernel_initializer='he_uniform')(X_actor)
            
            self.actor = Model(inputs = X_input_actor, outputs = Concatenate()([mu,sigma]))
            self.actor.compile(loss=self.continuous_actor_loss, optimizer=self.actor_optimizer)
        
        self.critic = Model(inputs = X_input_critic, outputs = value)
        self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
    
    def __init_buffers(self):
        self.buffer = ReplayBuffer()
        
    def log_pdf(self,mu, sigma, action):
        std = tf.clip_by_value(sigma, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
    
    def continuous_actor_loss(self, y_true, y_pred):
        actions, advantages = y_true[:, :1], y_true[:, 1:]
        mu,sigma = y_pred[:,:1], y_pred[:,1:]
        log_policy_pdf = self.log_pdf(mu,sigma,actions)
        loss_policy = log_policy_pdf * advantages
        
        return tf.reduce_sum(-loss_policy)

    def act(self,state):

        if self.action_space_mode == "discrete":
            prediction = self.actor.predict(state)[0]
            action = np.random.choice(self.n_actions, p=prediction)
            action_onehot = np.zeros([self.n_actions])
            action_onehot[action] = 1
        else:
            prediction = self.actor.predict(state)[0]
            mu = prediction[0]
            sigma = prediction[1]
            action = np.random.normal(mu, sigma,self.n_actions)
            action = np.clip(action, -self.action_bound, self.action_bound)
            action_onehot = action
        return action, action_onehot, prediction
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * self.gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        
        return discounted_r
    
    def replay(self):

        if self.buffer.size > 1:
            # reshape memory to appropriate shape for training
            states = np.vstack(self.buffer.states)
            actions = np.vstack(self.buffer.actions)

            # Compute discounted rewards
            discounted_r = self.discount_rewards(self.buffer.rewards)

            # Get Critic network predictions
            values = self.critic.predict(states)[:, 0]
            # Compute advantages
            advantages = discounted_r - values
            # training Actor and Critic networks


            if self.action_space_mode == "discrete":
                self.global_actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
            else:
                self.global_actor.fit(states,np.concatenate([actions,np.reshape(advantages,newshape=(len(advantages),1))],axis=1), epochs=1,verbose=0)

            self.global_critic.fit(states, discounted_r, epochs=1, verbose=0)
            
            # Reset weights
            self.actor.set_weights(self.global_actor.get_weights())
            self.critic.set_weights(
                self.global_critic.get_weights()
            )
            # reset training memory
            self.buffer.reset()
        

    def learn(self):
        global GLOBAL_EPISODE_NUM
        while self.max_episodes >= GLOBAL_EPISODE_NUM:
  
            score , done = 0, False

            state = self.env.reset()

            while not done:
                # self.env.render()
                state = np.expand_dims(state, axis=0)
                print(self.actor.predict(state))
                raise Exception('here')
                action, action_onehot, prediction = self.act(state)

                next_state, reward, done, _ = self.env.step(action)
                score += reward
                self.buffer.remember(state, action_onehot, reward)

    

                if self.buffer.size >= self.batch_size:
                    self.replay()


            # Episode ended
            self.running_reward.step(score)


            print(f"Episode#{GLOBAL_EPISODE_NUM}, Worker#{self.worker_id}, Reward:{episode_reward}")
            #tf.summary.scalar("episode_reward", score, step=GLOBAL_EPISODE_NUM)
            GLOBAL_EPISODE_NUM += 1

            # Else learn more
            self.replay()

    def run(self):
        state = self.env.reset()
        state = np.expand_dims(state, axis=0)
        print(self.critic(state))
        print()
        self.learn()


# In[9]:


from src.agents.agent import Agent
from src.utils.networks import CommonLayer
    

class A3CAgent(Agent):
    def __init__(self,
        environment,
        gamma = 0.99,
        policy="mlp",
        actor_optimizer=RMSprop(0.0001),
        critic_optimizer=RMSprop(0.0001),
        std_bound = [1e-2, 1.0],
        batch_size=64,
        n_workers=cpu_count()
    ):
        
        super(A3CAgent, self).__init__(environment,args=locals())
        
        global GLOBAL_EPISODE_NUM 
        GLOBAL_EPISODE_NUM += 1
        # Args
        self.environment = environment
        self.gamma = gamma
        self.std_bound = std_bound
        self.batch_size = batch_size
        self.policy = policy 
        self.actor_optimizer=actor_optimizer
        self.critic_optimizer=critic_optimizer
        self.n_workers = n_workers

        # Bootstrap
        self.__init_networks()
        self.__init_buffers()
        self._add_models_to_config([self.global_actor,self.global_critic])
        
    def __init_networks(self):
        X_input = Input(shape=self.observation_shape) 
        X = CommonLayer(X_input,self.policy)
        
        action = Dense(self.n_actions, activation="softmax", kernel_initializer='he_uniform')(X)
        value = Dense(1, kernel_initializer='he_uniform')(X)
        
        if self.action_space_mode == "discrete":
            action = Dense(self.n_actions, activation="softmax", kernel_initializer='he_uniform')(X)
            self.global_actor = Model(inputs = X_input, outputs = action)
            self.global_actor.compile(loss='categorical_crossentropy', optimizer=self.actor_optimizer)
        else:
            mu = Dense(self.n_actions, activation="tanh", kernel_initializer='he_uniform')(X)
            mu = Lambda(lambda x: x * self.action_bound)(mu)
            sigma = Dense(self.n_actions, activation="softplus", kernel_initializer='he_uniform')(X)
            
            self.global_actor = Model(inputs = X_input, outputs = Concatenate()([mu,sigma]))
            self.global_actor.compile(loss=self.continuous_actor_loss, optimizer=self.actor_optimizer)
        
        self.global_critic = Model(inputs = X_input, outputs = value)
        self.global_critic.compile(loss='mse', optimizer=self.critic_optimizer)
    
    def __init_buffers(self):
        self.buffer = ReplayBuffer()
        
    def log_pdf(self,mu, sigma, action):
        std = tf.clip_by_value(sigma, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
    
    def continuous_actor_loss(self, y_true, y_pred):
        actions, advantages = y_true[:, :1], y_true[:, 1:]
        mu,sigma = y_pred[:,:1], y_pred[:,1:]
        log_policy_pdf = self.log_pdf(mu,sigma,actions)
        loss_policy = log_policy_pdf * advantages
        
        return tf.reduce_sum(-loss_policy)

    

    def learn(self, timesteps=-1, plot_results=True, reset=False, success_threshold=False, log_level=1, log_each_n_episodes=50,max_episodes=10000):
        workers = []

        for i in range(self.n_workers):
            env = self.environment()
            workers.append(
                A3CWorker(
                    i,
                    self.env, 
                    self.global_actor, 
                    self.global_critic, 
                    self.action_space_mode,
                    self.observation_shape,
                    self.policy,
                    self.n_actions,
                    self.actor_optimizer,
                    self.critic_optimizer,
                    self.std_bound,
                    max_episodes
                )
            )

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


# In[10]:


from src.environments.discrete.cartpole import environment
agent = A3CAgent(environment, n_workers=1)
agent.learn()


# In[7]:


#from src.environments.continuous.inverted_pendulum import environment
#agent = A2CAgent(environment)
#agent.learn()



# In[ ]:




