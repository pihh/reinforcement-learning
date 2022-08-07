import numpy as np

from src.agents.agent import Agent
from src.utils.buffer import Buffer

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate



from utils.noise import OUActionNoise 

class Buffer:
    def __init__(self,n_actions, n_states, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, n_states))
        self.action_buffer = np.zeros((self.buffer_capacity, n_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, n_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1


class DdpgAgent(Agent):
    def __init__(self,
                 environment,
                 gamma = 0.99,
                 tau= 0.005,
                 std_dev = 0.2,
                 critic_lr = 0.002,
                 actor_lr = 0.001,
                 buffer_size=50000,
                 batch_size=64,
                 critic_optimizer = tf.keras.optimizers.Adam,
                 actor_optimizer = tf.keras.optimizers.Adam,
        ):
        super(DdpgAgent,self).__init__(environment,loss_keys=["actor_loss","critic_loss"],args=locals())
        
        self.std_dev = std_dev
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        self.noise = OUActionNoise(mean=np.zeros(self.n_actions), std_deviation=float(std_dev) * np.ones(self.n_actions))
        
        self.critic_optimizer = critic_optimizer(critic_lr)
        self.actor_optimizer = actor_optimizer(actor_lr)

        # Discount factor for future rewards
        self.gamma = gamma
        # Used to update target networks
        self.tau = tau

        if self.action_space_mode != "continuous":
            raise Exception('DDPG only accepts continuous action spaces')

        self.__init_networks()
        self.__init_buffers()

        self._add_models_to_config([self.actor,self.target_actor,self.critic,self.target_critic])
        self._init_tensorboard()
        
    def __init_buffers(self):
        self.buffer = Buffer(self.n_actions, self.n_inputs, self.buffer_size, self.batch_size)
            
    def __init_networks(self):
        
        def create_actor():
            # Initialize weights between -3e-3 and 3-e3
            last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

            inputs = Input(shape=self.env.observation_space.shape)
            out = Flatten()(inputs)
            out = Dense(256, activation="relu")(out)
            out = Dense(256, activation="relu")(out)
            outputs = Dense(self.n_actions, activation="tanh", kernel_initializer=last_init)(out)

            # Our upper bound is 2.0 for Pendulum.
            outputs = outputs * self.action_upper_bounds
            return tf.keras.Model(inputs, outputs)
        
        def create_critic():
            # State as input
            state_input = Input(shape=self.env.observation_space.shape)
            state_out = Flatten()(state_input)
            state_out = Dense(16, activation="relu")(state_out)
            state_out = Dense(32, activation="relu")(state_out)

            # Action as input
            action_input = Input(shape=(self.n_actions))
            action_out = Dense(32, activation="relu")(action_input)

            # Both are passed through seperate layer before concatenating
            concat = Concatenate()([state_out, action_out])

            out = Dense(256, activation="relu")(concat)
            out = Dense(256, activation="relu")(out)
            outputs = Dense(1)(out)

            # Outputs single value for give state-action
            return tf.keras.Model([state_input, action_input], outputs)
        
        self.actor = create_actor()
        self.target_actor = create_actor()
        
        self.critic = create_critic()
        self.target_critic = create_critic()
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
    
    def choose_action(self,state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.actor(state))
        noise = self.noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.action_lower_bounds, self.action_upper_bounds)

        return [np.squeeze(legal_action)]
    
    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
            
    @tf.function
    def update(self,state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

        self.learning_log.step_loss({
            "actor_loss":tf.get_static_value(actor_loss),
            "critic_loss":tf.get_static_value(critic_loss)
        })

        self.write_tensorboard_scaler('actor_loss',tf.get_static_value(actor_loss),self.learning_log.learning_steps)
        self.write_tensorboard_scaler('critic_loss',tf.get_static_value(critic_loss),self.learning_log.learning_steps)

            
    def replay(self):

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
        
    def test(self, episodes=10, render=True):

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
                    
                action = self.choose_action(state)

                # Step
                state,reward,done, info = self.env.step(action)

                # Get next state
                score += reward
            
            if render:
                self.env.close()

            print("Test episode: {}, score: {:.2f}".format(episode,score)) 
    
    def learn(self, timesteps=-1, plot_results=True, reset=True,  success_threshold=False,log_level=1, log_every=50,success_threshold_lookback=100):
        success_threshold = self.on_learn_start(timesteps,success_threshold,reset,success_threshold_lookback)

        score = 0
        timestep = 0
        episode = 0
        while self.learning_condition(timesteps,timestep):  # Run until solved
            prev_state = self.env.reset()
            score = 0
            done = False
            while not done:
                
                action = self.choose_action(prev_state)
                state, reward, done, info = self.env.step(action)
                self.buffer.record((prev_state, action, reward, state))
                self.replay()
                self.update_target(self.target_actor.variables, self.actor.variables, self.tau)
                self.update_target(self.target_critic.variables, self.critic.variables, self.tau)
                prev_state=state
                score += reward
                timestep+=1
                         
            # Episode done

            # track rewards, log , write to tensorboard
            self.on_learn_episode_end(score,log_every,log_level,success_threshold)
            
            # Log details
            episode += 1

            if self.did_finnish_learning(success_threshold,episode):
                break


        if plot_results:
            self.plot_learning_results()