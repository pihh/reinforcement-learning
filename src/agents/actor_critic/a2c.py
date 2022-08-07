import shutup
shutup.please()

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Model

from tensorflow.python.framework.ops import disable_eager_execution

from src.agents.agent import Agent
from src.utils.networks import CommonLayer

disable_eager_execution()

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

class ActorNetwork():
    def __init__(self,
                 observation_shape,
                 action_space_mode,
                 action_bound,
                 policy,
                 n_actions, 
                 optimizer=Adam,
                 learning_rate=0.01,
                 std_bound = [1e-2, 1.0],
    ):
        
        self.observation_shape = observation_shape
        self.policy = policy
        self.n_actions = n_actions
        self.action_space_mode = action_space_mode
        self.std_bound=std_bound
        self.action_bound = action_bound
        
        optimizer = optimizer(learning_rate)
        
        X_input = Input(shape=self.observation_shape) 
        X = CommonLayer(X_input,self.policy)
        
        if self.action_space_mode == "discrete":
            action = Dense(self.n_actions, activation="softmax", kernel_initializer='he_uniform')(X)
            self.model = Model(inputs = X_input, outputs = action)
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        else:
            mu = Dense(self.n_actions, activation="tanh", kernel_initializer='he_uniform')(X)
            mu = Lambda(lambda x: x * self.action_bound)(mu)
            sigma = Dense(self.n_actions, activation="softplus", kernel_initializer='he_uniform')(X)
            
            self.model = Model(inputs = X_input, outputs = Concatenate()([mu,sigma]))
            self.model.compile(loss=self.continuous_actor_loss, optimizer=optimizer)
    
    def log_pdf(self,mu, sigma, action):
        std = tf.clip_by_value(sigma, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
    
    def continuous_actor_loss(self, y_true, y_pred):
        actions, advantages = y_true[:, :self.n_actions], y_true[:, self.n_actions:]
        mu,sigma = y_pred[:,:1], y_pred[:,1:]
        log_policy_pdf = self.log_pdf(mu,sigma,actions)
        loss_policy = log_policy_pdf * advantages
        
        return tf.reduce_sum(-loss_policy)
    
    def act(self,state):
        state = np.expand_dims(state, axis=0)
        if self.action_space_mode == "discrete":
            prediction = self.model.predict(state)[0]
            action = np.random.choice(self.n_actions, p=prediction)
            action_onehot = np.zeros([self.n_actions])
            action_onehot[action] = 1
        else:
            prediction = self.model.predict(state)[0]
            mu = prediction[0]
            sigma = prediction[1]
            action = np.random.normal(mu, sigma,self.n_actions)
            action = np.clip(action, -self.action_bound, self.action_bound)
            action_onehot = action
        return action, action_onehot, prediction
    
class CriticNetwork():
    def __init__(self,
                 observation_shape,
                 action_space_mode,
                 policy,
                 n_actions, 
                 optimizer=Adam,
                 learning_rate=0.01,
                 std_bound = [1e-2, 1.0],
    ):

        self.observation_shape = observation_shape
        self.policy = policy
        self.n_actions = n_actions
        self.action_space_mode = action_space_mode
        self.std_bound=std_bound
        
        optimizer = optimizer(learning_rate)
        
        X_input = Input(shape=self.observation_shape) 
        X = CommonLayer(X_input,self.policy)
        
        value = Dense(1, kernel_initializer='he_uniform')(X)
        
        self.model = Model(inputs = X_input, outputs = value)
        self.model.compile(loss='mse', optimizer=optimizer)

class A2CAgent(Agent):
    def __init__(self,
                environment,
                gamma = 0.99,
                policy="mlp",
                actor_optimizer=RMSprop,
                critic_optimizer=RMSprop,
                actor_learning_rate=0.001,
                critic_learning_rate=0.001,
                std_bound = [1e-2, 1.0],
                batch_size=64,
                epochs=1
                ):
        super(A2CAgent, self).__init__(environment,args=locals())
        
        # Args
        self.gamma = gamma
        self.std_bound = std_bound
        self.batch_size = batch_size
        self.policy = policy 
        self.actor_optimizer=actor_optimizer
        self.critic_optimizer=critic_optimizer
        self.actor_learning_rate= actor_learning_rate
        self.critic_learning_rate=critic_learning_rate
        self.epochs=epochs

        # Bootstrap
        self.__init_networks()
        self.__init_buffers()
        self._add_models_to_config([self.actor.model,self.critic.model])
        self._init_tensorboard()
        
    def __init_networks(self):
        self.actor = ActorNetwork(
            observation_shape=self.observation_shape,
            action_space_mode=self.action_space_mode,
            policy=self.policy,
            n_actions=self.n_actions, 
            optimizer=self.actor_optimizer,
            learning_rate=self.actor_learning_rate,
            std_bound = self.std_bound,
            action_bound = self.action_bound
        )
        
        self.critic = CriticNetwork(
            observation_shape=self.observation_shape,
            action_space_mode=self.action_space_mode,
            policy=self.policy,
            n_actions=self.n_actions, 
            optimizer=self.critic_optimizer,
            learning_rate=self.critic_learning_rate,
            std_bound = self.std_bound
        )
    
    def __init_buffers(self):
        self.buffer = ReplayBuffer()
        
    def act(self,state):
        action, action_onehot, prediction = self.actor.act(state)
        return action, action_onehot, prediction
    
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
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
            values = self.critic.model.predict(states)[:, 0]
            # Compute advantages
            advantages = discounted_r - values
            # training Actor and Critic networks


            if self.action_space_mode == "discrete":
                self.actor.model.fit(states, actions, sample_weight=advantages, epochs=self.epochs, verbose=0)
            else:
                self.actor.model.fit(states,np.concatenate([actions,np.reshape(advantages,newshape=(len(advantages),1))],axis=1), epochs=self.epochs,verbose=0)

            self.critic.model.fit(states, discounted_r, epochs=self.epochs, verbose=0)
            # reset training memory
            self.buffer.reset()

    def save(self):
        self.actor.model.save_weights('a2c-actor_'+self.hash)
        self.critic.model.save_weights('a2c-critic_'+self.hash)
        
    def learn(self, timesteps=-1, plot_results=True, reset=True, success_threshold=False, log_level=1, log_every=50 , success_threshold_lookback=100 , success_strict=False):
        

        #self.validate_learn(timesteps,success_threshold,reset)
        #success_threshold = success_threshold if success_threshold else self.env.success_threshold

        success_threshold = self.on_learn_start(timesteps,success_threshold,reset,success_threshold_lookback, success_strict)

        timestep = 0
        episode = 0
        
        while self.learning_condition(timesteps,timestep):  # Run until solved
            state = self.env.reset()
            score = 0
            done = False
            
            while not done:
                
                #state = np.expand_dims(state, axis=0)
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                # Memorize (state, action, reward) for training
                self.buffer.remember(np.expand_dims(state, axis=0), action_onehot, reward)
                # Update current state
                state = next_state
                score += reward
                timestep +=1
                
                if self.buffer.size >= self.batch_size:
                    self.replay()
            
            # Episode ended
            episode += 1

            # Step reward, tensorboard log score, print progress
            self.on_learn_episode_end(score,log_every,log_level,success_threshold)
            
            # If done stop
            if self.did_finnish_learning(success_threshold,episode):
                break
                
            # Else learn more
            #self.replay()
        
        # End of trainig
        self.env.close()
        
        if plot_results:
            self.plot_learning_results()

