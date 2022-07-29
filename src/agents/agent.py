import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from src.utils.running_reward import RunningReward
from src.utils.gym_environment import GymEnvironment
from src.utils.logger import LearningLogger
from src.utils.tensorboard_writer import create_writer, graph, histogram, scalar

class Agent:
    def __init__(self,
        environment,
        loss_keys=[],
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.00001,
        **kwargs
    ):
        # Args
        self._environment = environment
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_log_loss_keys = loss_keys

        self.epsilon_ = epsilon
        self.epsilon_decay_ = epsilon_decay
        
        # Boot
        self.__init_environment()
        self.__init_reward_tracker()
        self.__init_config_hash(kwargs)
        self.__init_loggers()

    def __init_config_hash(self,args={}):
        config = {}
        if 'args' in args:
            args = args['args']
            del args['self']
            del args['environment'] 
            del args['__class__']
            for arg in args:
                t = type(args[arg])
                if t not in [int,float,str,bool]:
                    config[arg] = t.__name__
                else:
                    config[arg] = args[arg]
                if isinstance(config[arg], np.ndarray):
                    config[arg] = config[arg].tolist()


        #     print()
        #     print('config before',config)
        #     print()
        # print('init_loggers')
 
        keys = self.__dict__.copy()
        del keys['_environment']
        del keys['running_reward']
        del keys['env']
        for key in keys:
            config[key] = keys[key]
            if isinstance(config[key], np.ndarray):
                config[key] = config[key].tolist()

        config['env_name'] = self.env.spec.name
        config['agent'] = type(self).__name__

        self.config = config
        
    def _add_models_to_config(self,models):
        self.config['models'] = {}
        for model in models:
            self.config['models'][model.name] = []
            for layer in model.layers:
                name = type(layer).__name__
                try:
                    activation = layer.activation.__name__
                except:
                    activation = None 
                try:
                    units = layer.units
                except:
                    units = None 
                try:
                    kernel_initializer = layer.kernel_initializer.__name__
                except: 
                    kernel_initializer = None 
            
                self.config['models'][model.name].append({
                    "name":name,
                    "activation":activation,
                    "units":units,
                    "kernel_initializer":kernel_initializer
                })



    def _init_tensorboard(self):
        self.hash = hashlib.md5(json.dumps(self.config,sort_keys=True, indent=2).encode('utf-8')).hexdigest()
        self.tensorboard_writer, self.tensorboard_writer_log_directory = create_writer(self.config['env_name'],self.config['agent'],self.hash)   
        with open(self.tensorboard_writer_log_directory+'/config.json', 'w') as f:
            json.dump(self.config, f, indent=2)


    def __init_loggers(self):
        self.learning_log = LearningLogger(self.learning_log_loss_keys)

            
    def __init_environment(self):
        env = GymEnvironment(self._environment)
        self.env = env.env
        self.n_actions = env.n_actions
        self.n_inputs = env.n_inputs
        self.actions = env.actions
        self.observation_shape = env.observation_shape
        self.action_space_mode = env.action_space_mode
        self.action_upper_bounds = env.action_upper_bounds
        self.action_lower_bounds = env.action_lower_bounds
        self.action_bound = env.action_bound

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

    # lifecycle hooks 
    def on_learn_start(self,timesteps,success_threshold,reset):
        self.validate_learn(timesteps,success_threshold,reset)

        return success_threshold if success_threshold else self.env.success_threshold

    def on_learn_end(self,plot_results):
        if plot_results:
            self.plot_learning_results()

    def after_learn_cycle(self):
        pass 

    def on_learn_episode_end(self,score,log_each_n_episodes,log_level,success_threshold,):
        self.running_reward.step(score)
    
        self.learning_log.episode(
            log_each_n_episodes,
            score,
            self.running_reward.reward, 
            log_level=log_level
        )
            
        return self.did_finnish_learning(success_threshold,self.running_reward.episodes)
            

    def on_test_episode_start(self):
        try:
            state = self.env.reset()
        except:
            # Hack pybullet envs
            self._Agent__init_environment()
            state = self.env.reset()
        
        return state

    def on_test_episode_end(self,episode,score,render):
        if render:
            self.env.close()
        self.learning_log.episode_test_log(score,episode)
        
    def before_learn_cycle(self):
        pass 

    def before_learn_episode(self):
        pass

    def before_test_episode(self):
        pass


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


    def write_tensorboard_scaler(self, scalar_name,score, steps):
        scalar(self, scalar_name,score, steps)

    def write_tensorboard_histogram(self, name, model):
        histogram(self, name, model)

    def decrement_epsilon(self):
        #self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > 0.01 else 0.01
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def plot_learning_results(self):
        plt.figure(figsize=(16,4))
        plt.plot(self.running_reward.reward_history)
        plt.show()
        
    def choose_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
