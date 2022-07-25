import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.state_memory = np.zeros((self.buffer_size, *input_shape))
        self.new_state_memory = np.zeros((self.buffer_size, *input_shape))
        self.action_memory = np.zeros((self.buffer_size, n_actions))
        self.reward_memory = np.zeros(self.buffer_size)
        self.done_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.buffer_counter % self.buffer_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done

        self.buffer_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.buffer_counter, self.buffer_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return states, actions, rewards, states_, dones

from src.agents.agent import Agent


class SoftActorCriticAgent(Agent):
    def __init__(self, 
            environment,
            alpha=0.0003, 
            beta=0.0003, 
            gamma=0.99, 
            tau=0.005,
            buffer_size=1000000, 
            policy="mlp", 
            batch_size=256, 
            reward_scale=2, 
            loss_function = keras.losses.MSE, #keras.losses.Huber()
    ):
        super(SoftActorCriticAgent, self).__init__(environment)
        
        print(self.env,self.n_actions)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.policy = policy
        self.reward_scale = reward_scale
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.loss_function = loss_function
        
        self.__init_networks()
        self.__init_buffers()
        
    def __init_buffers(self):
        self.buffer = ReplayBuffer(self.buffer_size, self.observation_shape, self.n_actions)
            
    def __init_networks(self):
        self.actor = ActorNetwork(n_actions=self.n_actions,policy=self.policy, max_action=self.env.action_space.high)
        self.critic_1 = CriticNetwork(n_actions=self.n_actions,policy=self.policy, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=self.n_actions,policy=self.policy, name='critic_2')
        self.value = ValueNetwork(name='value',policy=self.policy)
        self.target_value = ValueNetwork(name='target_value',policy=self.policy)

        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=self.beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=self.beta))
        self.value.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_value.compile(optimizer=Adam(learning_rate=self.beta))

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.remember(state, action, reward, new_state, done)      

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
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        
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
        self.value.optimizar.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))
        
        # Actor network update
        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,reparameterize=True)
            
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
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

        self.update_network_parameters()