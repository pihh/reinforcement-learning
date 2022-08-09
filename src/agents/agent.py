import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d

from src.utils.running_reward import RunningReward
from src.utils.gym_environment import GymEnvironment
from src.utils.logger import LearningLogger
from src.utils.writer import ResultsWriter, create_writer, graph, histogram, log_environment_results_file, scalar

class Agent:
    def __init__(self,
        environment,
        loss_keys=[],
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.00001,
        success_threshold_lookback=100,
        **kwargs
    ):
        # Args
        self._environment = environment
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_log_loss_keys = loss_keys
        self.success_threshold_lookback=success_threshold_lookback

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
                    try:
                        config[arg] = args[arg].__name__
                    except:
                        config[arg] = t.__name__
                else:
                    config[arg] = str(args[arg])
                if isinstance(config[arg], np.ndarray):
                    config[arg] = config[arg].tolist()



 
        keys = self.__dict__.copy()
        del keys['_environment']
        del keys['running_reward']
        del keys['env']
        for key in keys:
            config[key] = keys[key]
            if isinstance(config[key], np.ndarray):
                config[key] = config[key].tolist()

            config[key] = str(config[key])

        
        config['env_name'] = self.env.env_name if hasattr(self.env,'env_name') else self.env.spec.name
        config['agent'] = type(self).__name__

        if hasattr(self.env,'config'):
            config['env_config'] = self.env.config

        self.config = config

    def _save_weights(self,model_list):
        success = True
        for model in model_list:
            try:
                model_name = model['name']
                model_instance = model['model']
                model_instance.save_weights(self.writer_log_directory+'/models/'+model_name+'.h5')
                
            except Exception as e:
                success = False
                print(e)
        
        if success == True:
            print('* Models saved *')

    def _load_weights(self,model_list):
        success = True
        for model in model_list:
            try:
                model_name = model['name']
                model_instance = model['model']
                model_instance.load_weights(self.writer_log_directory+'/models/'+model_name+'.h5')
                self._load_best_score()
       
            except Exception as e:
                success = False
                print(e)
                print()
        
        if success == True:
            print('* Models successfully loaded *')

    def _load_best_score(self):
        score = self.results_writer.load_best_score()
        if score != False:
            self.learning_max_score = score

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

        config = self.config.copy()
        self.hash = hashlib.md5(json.dumps(config,sort_keys=True, indent=2).encode('utf-8')).hexdigest()
        self.writer, self.writer_log_directory = create_writer(self.config['env_name'],self.hash)   

        with open(self.writer_log_directory+'/config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        self.results_writer = ResultsWriter(self.config['env_name'],self.hash)

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
        self.running_reward = RunningReward(success_threshold_lookback=self.success_threshold_lookback)
        
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
        if episode > self.success_threshold_lookback:
            if self.running_reward.moving_average > success_threshold:
                print('Agent solved environment at the episode {}'.format(episode))
                return True
            elif self.success_strict != False:
                if np.all(np.array(self.running_reward.reward_history[-self.running_reward.success_threshold_lookback:])>0):
                    print('All episodes are returning positive profits.Agent solved environment at the episode {}'.format(episode))
                    return True
            return False
        return False

    # lifecycle hooks 
    def on_learn_start(self,timesteps,success_threshold,reset,success_threshold_lookback=100, success_strict=False):
        self.learning_max_score = 0
        self.success_strict = success_strict

        self.validate_learn(timesteps,success_threshold,reset)

        self.running_reward.success_threshold_lookback = success_threshold_lookback
        self.learning_log.success_threshold_lookback = success_threshold_lookback
        self.success_threshold_lookback = success_threshold_lookback

        return success_threshold if success_threshold else self.env.success_threshold

    def on_learn_episode_end(self,score,log_every,log_level,success_threshold):
        self.running_reward.step(score)
    
        self.learning_log.episode(
            log_every,
            score,
            self.running_reward.moving_average, 
            log_level=log_level
        )
        try:
            if not np.isinf(self.running_reward.moving_average):
                self.write_tensorboard_scaler('episode_score',score, self.running_reward.episodes)
                self.write_tensorboard_scaler('episode_score_moving_average_'+str(self.running_reward.success_threshold_lookback),self.running_reward.moving_average, self.running_reward.episodes)
        except:
            pass

        if self.running_reward.episodes > self.success_threshold_lookback:
            if self.running_reward.moving_average > self.learning_max_score:
                self.learning_max_score = self.running_reward.moving_average
                try:
                    self.save()
                    print()
                    print('New historical moving average record: {:.5f}'.format(self.learning_max_score))
                    print()
                except:
                    print('Failed to save')

                try:
                    self.results_writer.log(self.learning_max_score)
                except:
                    print('Failed to write results log')

            
        return self.did_finnish_learning(success_threshold,self.running_reward.episodes)
            
    def on_learn_end(self,plot_results):
        self.env.close()

        if plot_results:
            self.plot_learning_results()

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
        ysmoothed = gaussian_filter1d(self.running_reward.reward_history, sigma=10)

        plt.figure(figsize=(16,4))
        plt.plot(self.running_reward.reward_history, alpha=0.75)
        plt.plot(np.zeros(len(self.running_reward.reward_history)), color="black", alpha=0.5)
        plt.plot(ysmoothed, color="red")

        plt.show()

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
