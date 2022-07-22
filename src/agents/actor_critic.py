import itertools
import numpy as np

from src.agents.agent import Agent
from src.utils.buffer import Buffer

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from tensorflow.keras.layers import Input, Dense, Concatenate

class ActorCriticAgent(Agent):
    def __init__(self, 
                environment, 
                alpha = 0.01,
                gamma = 0.99,
                eps = np.finfo(np.float32).eps.item(),
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
                critic_loss= tf.keras.losses.Huber()):
        
        super(ActorCriticAgent, self).__init__(environment)
        
        # Args
        self.alpha = alpha
        self.gamma = gamma 
        self.eps = eps 
        self.optimizer=optimizer
        self.critic_loss = critic_loss
        #type(tf.keras.optimizers.Adam(learning_rate=0.01)).__name__

        self.__init_networks()
        self.__init_buffers()
        
    def __init_buffers(self):
        self.buffer = Buffer(['action_log_probs','critic_values','rewards'])
            
    def __init_networks(self):
        num_inputs = self.observation_shape[0]
        num_hidden = 128

        inputs = Input(shape=(num_inputs,),name="actor_critic_inputs")
        common_layer = Dense(num_hidden, activation="relu", name="actor_critic_common_layer")(inputs)
        
        if self.action_space_mode == "discrete":
            action = Dense(self.n_actions, activation="softmax")(common_layer)
        elif self.action_space_mode == "continuous":
            mu_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            mu = Dense(self.n_actions, activation="tanh" , name='mu',kernel_initializer=mu_init)(common_layer)
            
            sigma_init = tf.random_uniform_initializer(minval=self.eps, maxval=0.2)
            sigma = Dense(self.n_actions, activation="softplus", name="sigma", kernel_initializer=sigma_init)(common_layer)
            
            action = Concatenate(axis=-1, name="actor_output")([mu,sigma]) * self.action_upper_bounds

        critic = Dense(1)(common_layer)

        self.model = keras.Model(inputs=inputs, outputs=[action, critic])

    def choose_action(self, state, deterministic=True):
        action_probs, critic_value = self.model(state)
        
        if self.action_space_mode == "discrete":
            # DISCRETE SAMPLING
            if deterministic:
                action = np.argmax(np.squeeze(action_probs))
                action_log_prob = action
            else:
                # Sample action from action probability distribution
                action = np.random.choice(self.n_actions, p=np.squeeze(action_probs))
                action_log_prob = tf.math.log(action_probs[0, action])


        elif self.action_space_mode == "continuous":
            # CONTINUOUS SAMPLING
            mu = action_probs[:,0:self.n_actions]
            sigma = action_probs[:,self.n_actions:]
            
            if deterministic:
                action = mu
                action_log_prob = action
            else:
                norm_dist = tfp.distributions.Normal(mu, sigma)
                action = tf.squeeze(norm_dist.sample(self.n_actions), axis=0)
                action_log_prob = -(norm_dist.log_prob(action)+self.eps)
                action = tf.clip_by_value(
                    action, self.env.action_space.low[0], 
                    self.env.action_space.high[0])

                action = np.array(action[0],dtype=np.float32)
        
        return action, action_log_prob , critic_value
    
    def test(self, episodes=10, render=True):

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if render:
                    self.env.render()
                
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)
                
                # Sample action, probs and critic
                action, action_log_prob, critic_value = self.choose_action(state)

                # Step
                state,reward,done, info = self.env.step(action)

                # Get next state
                score += reward
            
            if render:
                self.env.close()

            print("Test episode: {}, score: {:.2f}".format(episode,score))
    
    def learn(self, timesteps=-1, plot_results=True, reset=False, log_each_n_episodes=100, success_threshold=False):
        
        self.validate_learn(timesteps,success_threshold,reset)
        success_threshold = success_threshold if success_threshold else self.env.success_threshold
 
        self.buffer.reset()
        score = 0
        timestep = 0
        episode = 0
        
        while self.learning_condition(timesteps,timestep):  # Run until solved
            state = self.env.reset()
            score = 0
            done = False
            with tf.GradientTape() as tape:
                while not done:

                    state = tf.convert_to_tensor(state)
                    state = tf.expand_dims(state, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    action, action_log_prob, critic_value = self.choose_action(state, deterministic=False)

                    self.buffer.store('critic_values',critic_value[0, 0])

                    # Sample action from action probability distribution
                    self.buffer.store('action_log_probs',action_log_prob)

                    # Apply the sampled action in our environment
                    state, reward, done, _ = self.env.step(action)
                    self.buffer.store('rewards',reward)

                    score += reward
                    timestep+=1
                # Update running reward to check condition for solving
                self.running_reward.step(score)

                # Time discounted rewards
                returns = []
                discounted_sum = 0
                
                for r in self.buffer.get('rewards')[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(self.buffer.get('action_log_probs'), self.buffer.get('critic_values'), returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        self.critic_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                #print(loss_value)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Clear the loss and reward history
                self.buffer.reset()

            # Log details
            episode += 1
            if episode % log_each_n_episodes == 0 and episode > 0:
                print('episode {}, running reward: {:.2f}, last reward: {:.2f}'.format(episode,self.running_reward.reward, score))

            if self.did_finnish_learning(success_threshold,episode):
                    break

        if plot_results:
            self.plot_learning_results()



