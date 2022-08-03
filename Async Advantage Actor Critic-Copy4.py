# Tutorial by www.pylessons.com
# Tutorial written for - Tensorflow 2.3.1

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)

    #X = Conv2D(32, 8, strides=(4, 4),padding="valid", activation="elu", data_format="channels_first", input_shape=input_shape)(X_input)
    #X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="elu", data_format="channels_first")(X)
    #X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="elu", data_format="channels_first")(X)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))

    return Actor, Critic

class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 20000, 0, -21.0 # specific for pong
        self.lock = Lock()
        self.lr = 0.001

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = self.env.observation_space.shape
        #self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        actions = np.vstack(actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)

        # Get Critic network predictions
        value = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - value
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

        #actor.set_weights(self.Actor.get_weights())
        #critic.set_weights(self.Critic.get_weights())

        #return actor,critic
 
    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')


    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        return next_state, reward, done, info
    
    def run(self):
        scores=[]
        for e in range(self.EPISODES):
            state = self.env.reset()
            done, score, SAVING = False, 0, ''
            # Instantiate or reset games memory
            states, actions, rewards = [], [], []
            while not done:
                #self.env.render()
                # Actor picks an action
                action = self.act(np.expand_dims(state,axis=0))
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                # Memorize (state, action, reward) for training
                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    # average = self.PlotModel(score, e)
                    # # saving best models
                    # if average >= self.max_average:
                    #     self.max_average = average
                    #     self.save()
                    #     SAVING = "SAVING"
                    # else:
                    #     SAVING = ""
                    # print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                    scores.append(score)
                    if self.episode % 10:
                        print("episode: {}/{},  score: {}, average: {:.2f} ".format(self.episode, self.EPISODES, score, np.mean(scores[-100:]) ) )
                    self.replay(states, actions, rewards)
                    self.episode +=1
         # close environemnt when finish training  
        self.env.close()

    def train(self, n_threads):
        self.scores = []
        self.env.close()
        # Instantiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]
        
        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self,
                    envs[i],
                    i,
                    OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr))) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
            
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print('exiting')
            
    def train_threading(self, agent, env, thread, models):
        actor,critic = models
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = env.reset()
            # Instantiate or reset games memory
            states, actions, rewards = [], [], []
            while not done:
                expanded_state = np.expand_dims(state,axis=0)
                prediction = agent.Actor.predict(expanded_state)[0]
                action = np.random.choice(self.action_size, p=prediction)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                
                score += reward
                state = next_state

            self.lock.acquire()
            self.scores.append(score)
            self.replay(states, actions, rewards)
            self.lock.release()
                    
            # Update episode count
            with self.lock:
                # average = self.PlotModel(score, self.episode)
                # # saving best models
                # if average >= self.max_average:
                #     self.max_average = average
                #     self.save()
                #     SAVING = "SAVING"
                # else:
                #     SAVING = ""
                if self.episode % 10 == 0:
                    print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} ".format(self.episode, self.EPISODES, thread, score, np.mean(self.scores[-100:])))
                if(self.episode < self.EPISODES):
                    self.episode += 1
        env.close()            

    # def test(self, Actor_name, Critic_name):
    #     self.load(Actor_name, Critic_name)
    #     for e in range(100):
    #         state = self.reset(self.env)
    #         done = False
    #         score = 0
    #         while not done:
    #             self.env.render()
    #             action = np.argmax(self.Actor.predict(state))
    #             state, reward, done, _ = self.step(action, self.env, state)
    #             score += reward
    #             if done:
    #                 print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
    #                 break
    #     self.env.close()

if __name__ == "__main__":
    env_name = 'PongDeterministic-v4'
    env_name = 'LunarLander-v2'
    #env_name = 'Pong-v0'
    agent = A3CAgent(env_name)
    #agent.run() # use as A2C
    agent.train(n_threads=12) # use as A3C
    #agent.test('Models/Pong-v0_A3C_2.5e-05_Actor.h5', '')