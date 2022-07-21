import itertools
import numpy as np
import matplotlib.pyplot as plt

from src.utils.gym_environment import GymEnvironment

class QAgent:
    def __init__(self, 
                environment, 
                alpha = 0.1,
                gamma = 0.99,
                epsilon=1.0,
                epsilon_decay=0.996,
                bucket_size=20):

        # Args
        self.bucket_size = bucket_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Environment
        env = GymEnvironment(environment)
        self.env = env.env
        self.n_actions = env.n_actions
        self.actions = env.actions
        self.observation_shape = env.observation_shape
        
        self.__init_buckets()
        self.__init_q()

    def __init_buckets(self):

        individual_observations = self.observation_shape[0]
        self.buckets = []
        # in the MountainCar case, pos_space, vel_space 
        for i in range(individual_observations):
            self.buckets.append(
                np.linspace(
                    self.env.observation_space.low[i],
                    self.env.observation_space.high[i],
                    self.bucket_size
                )
            )
        
    def __init_q(self):
        Q = {}
        states = []
        actions = self.actions
        
        bucket_indices = []
        for bucket in self.buckets:
            bucket_indices.append(range(len(bucket)+1))
            
        for b in itertools.product(*bucket_indices):
            states.append(b)
        
        for state in states:
            for action in [0,1,2]:
                Q[state,action] = 0
                
        self.Q = Q

    def get_state(self,observation):
        #pos,vel = observation
        state = []
        for i in range(len(self.buckets)):
            state.append(np.digitize(observation[i],self.buckets[i]))

        return tuple(state)

    def max_action(self, state):
        values = np.array([self.Q[state,a] for a in self.actions])
        action = np.argmax(values)
        
        return action
    
    def decrement_eps(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > 0.01 else 0.01
    
    def learn(self, timesteps=-1, n_games=50000, success_threshold=-150, plot_results=True):
        
        obs = self.env.reset()
        state = self.get_state(obs)
        
        self.total_rewards = []
        self.avg_rewards = []
        
        score = 0
        timestep = 0
        episode = 0

        
        # Loop condition
        def learning_condition():
            if timesteps == -1:
                return True
            else: 
                return timesteps > timestep
        
        while learning_condition():
            
            # Choose action
            action = np.random.choice(self.actions) if np.random.random() < self.epsilon else self.max_action(state)
            
            # Step
            obs_,reward,done, info = self.env.step(action)
            
            # Get next state
            score += reward
            state_ = self.get_state(obs_)
            action_ = self.max_action(state_)

            # Update Q table
            self.Q[state,action] = self.Q[state,action] + self.alpha*(reward + self.gamma*self.Q[state_,action_] - self.Q[state,action])
            
            # Set state as next state so the agent keeps 
            state = state_
            
            if done:

                # Loop episode state
                if episode % 1000 == 0 and episode > 0:
                    print('episode',episode,'score',score,'epsilon %:.3f',self.epsilon)
                
                # Update pointers
                self.decrement_eps()
                self.total_rewards.append(score)
                
                # Track reward evolution
                if len(self.total_rewards) > 100:
                    avg_reward = np.mean(self.total_rewards[-100:])
                    self.avg_rewards.append(avg_reward)
                    
                    # Break loop if average reward is greater than success threshold
                    if avg_reward > success_threshold:
                        print('Agent solved environment at the episode {}'.format(episode))
                        break
                
                # Reset environment
                score = 0
                episode +=1
                obs = self.env.reset()
                state = self.get_state(obs)
                
            # Update timestep counter
            timestep+=1
        
        if plot_results:
            plt.plot(self.avg_rewards)
    