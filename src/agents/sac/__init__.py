import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

from src.agents.agent import Agent
from src.agents.sac.buffer import ReplayBuffer
from src.agents.sac.networks import CriticNetwork, ActorNetwork, ValueNetwork

class SoftActorCriticAgent(Agent):
    def __init__(self, 
            environment,
            alpha=0.0003, 
            beta=0.0003, 
            gamma=0.99, 
            delta=0.0003,
            tau=0.005,
            buffer_size=1000000, 
            policy="mlp", 
            batch_size=256, 
            reward_scale=2, 
            actor_optimizer= Adam,
            critic_optimizer=Adam,
            value_optimizer= Adam,
            loss_function = keras.losses.MSE, #keras.losses.Huber()
    ):
        super(SoftActorCriticAgent, self).__init__(environment,loss_keys=["actor","value","critic_1","critic_2"],args=locals())

        self._validate_continuous_action_space()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tau = tau
        self.policy = policy
        self.reward_scale = reward_scale
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.loss_function = loss_function

        self.actor_optimizer= actor_optimizer(learning_rate=self.alpha)
        self.critic_optimizer= critic_optimizer(learning_rate=self.beta)
        self.value_optimizer= value_optimizer(learning_rate=self.delta)
        
        self.__init_networks()
        self.__init_buffers()
        self._add_models_to_config([self.actor,self.critic_1,self.critic_2,self.value,self.target_value])
        self._init_tensorboard()
        
    def __init_buffers(self):
        self.buffer = ReplayBuffer(self.buffer_size, self.observation_shape, self.n_actions)
            
    def __init_networks(self):
        self.actor = ActorNetwork(n_actions=self.n_actions,policy=self.policy, max_action=self.env.action_space.high)
        self.critic_1 = CriticNetwork(n_actions=self.n_actions,policy=self.policy, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=self.n_actions,policy=self.policy, name='critic_2')
        self.value = ValueNetwork(name='value',policy=self.policy)
        self.target_value = ValueNetwork(name='target_value',policy=self.policy)

        self.actor.compile(optimizer=self.actor_optimizer)
        self.critic_1.compile(optimizer=self.critic_optimizer)
        self.critic_2.compile(optimizer=self.critic_optimizer)
        self.value.compile(optimizer=self.value_optimizer)
        self.target_value.compile(optimizer=self.value_optimizer)

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.buffer.remember(state, action, reward, new_state, done)      

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)
        
    def replay(self):
        if self.buffer.buffer_counter < self.batch_size:
            return
    
        state,action, reward, state_, done = self.buffer.sample(self.batch_size)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        # Value network update
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states),1)
            value_= tf.squeeze(self.target_value(states_),1)
            
            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs,1)
            
            q1_new_policy = self.critic_1(states,current_policy_actions)
            q2_new_policy = self.critic_2(states,current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy,q2_new_policy))
            
            value_target = critic_value - log_probs
            value_loss = 0.5 *self.loss_function(value,value_target)
            
            
        value_network_gradient = tape.gradient(value_loss,self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))
        
        # Actor network update
        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. 
            new_policy_actions, log_probs = self.actor.sample_normal(states,reparameterize=True)
            
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        # Critic network update
        with tf.GradientTape(persistent=True) as tape:
            
            q_hat = self.reward_scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * self.loss_function(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * self.loss_function(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

        self.learning_log.step_loss({
            "actor":actor_loss.numpy(),
            "value":value_loss.numpy(),
            "critic_1":critic_1_loss.numpy(),
            "critic_2":critic_2_loss.numpy()
        })

        # log evolution on tensorboard
        self.write_tensorboard_scaler('actor_loss',tf.get_static_value(actor_loss),self.learning_log.learning_steps)
        self.write_tensorboard_scaler('value_loss',tf.get_static_value(value_loss),self.learning_log.learning_steps)
        self.write_tensorboard_scaler('critic_1_loss',tf.get_static_value(critic_1_loss),self.learning_log.learning_steps)
        self.write_tensorboard_scaler('critic_2_loss',tf.get_static_value(critic_2_loss),self.learning_log.learning_steps)
        
        self.update_network_parameters()
        
    def test(self, episodes=10, render=True, init_environment=False):
        for episode in range(episodes):
            try:
                state = self.env.reset()
            except:
                self._Agent__init_environment()
                state = self.env.reset()
                
            done = False
            score = 0
            
            while not done:
                if render:
                    self.env.render()
                
                # Sample action, probs and critic
                action = self.choose_action(state)

                # Step
                state,reward,done, info = self.env.step(action)

                # Get next state
                score += reward
            
            if render:
                self.env.close()

            self.learning_log.episode_test_log(score,episode)
            
    def learn(self, timesteps=-1, plot_results=True, reset=True, success_threshold=False, log_level=1, log_every=50, success_threshold_lookback=100):
        
        success_threshold = self.on_learn_start(timesteps,success_threshold,reset,success_threshold_lookback)
 
        score = 0
        timestep = 0
        episode = 0
        
        while self.learning_condition(timesteps,timestep):  # Run until solved
            state = self.env.reset()
            score = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                score += reward
                self.remember(state, action, reward, state_, done)
                self.replay()
                state = state_
            
             # Log details
            episode += 1
            
            self.on_learn_episode_end(score,log_every,log_level,success_threshold)
           
            if self.did_finnish_learning(success_threshold,episode):
                break

        if plot_results:
            self.plot_learning_results()