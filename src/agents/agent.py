import json
import time
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d


from src.utils.gym_environment import GymEnvironment

from src.utils.running_reward import RunningReward

from src.utils.logger import LearningLogger

from src.utils.writer import ResultsWriter
from src.utils.writer import create_writer
from src.utils.writer import histogram
from src.utils.writer import scalar

from src.utils.helpers import get_number_of_files_in_folder
from src.utils.helpers import parse_time
from src.utils.helpers import md5

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
        """
        Creates Agent UUID to track it's configuration, results, and so on
        """
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

    def _validate_discrete_action_space(self):
        assert self.action_space_mode == "continuous", "This agent only works with discrete action spaces. Please select other agent for this task"

    def _validate_continuous_action_space(self):
        assert self.action_space_mode == "discrete", "This agent only works with continuous action spaces. Please select other agent for this task"

    def _save_weights(self,model_list):
        success = True
        failed_models = []
        for model in model_list:
            try:
                model_name = model['name']
                model_instance = model['model']
                model_instance.save_weights(self.writer_log_directory+'/models/'+model_name+'.h5')
                
            except Exception as e:
                failed_models.append(model['name'])
                success = False
                print(e)
        
        if success == True:
            print('* Models saved *')
        else:
            print()
            print('* Could not save the following models: {} *'.format(failed_models))
            print()

    def _load_weights(self,model_list):
        """
        Restore model weights and agent configuration
        """
        success = True
        failed_models = []
        for model in model_list:
            try:
                model_name = model['name']
                model_instance = model['model']
                model_instance.load_weights(self.writer_log_directory+'/models/'+model_name+'.h5')
                self._load_best_score()
       
            except Exception as e:
                failed_models.append(model['name'])
                success = False

        try:
            self._load_agent_configuration()
        except Exception as e:
            print('* Could not load the agent configuration * ')
            print()
            print(e)
            print()

        if success == False: 
            print()
            print('* Could not load the following models: {} *'.format(failed_models))
            print()

    def _load_best_score(self):
        """
        If has been trained before, load it's best score so it wont override the previous work
        """
        score = self.results_writer.load_best_score()
        if score != False:
            self.learning_max_score = score
            self.is_first_train = False
        else:
            self.learning_max_score = float(-np.inf)
            self.is_first_train = True


    def _load_agent_configuration(self):
        print('@TODO Agent._load_agent_configuration()')

    def _add_models_to_config(self,models):
        # To generate config.json and to restore in the future
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
        """
        Note: Should be renamed 
            * Creates agent unique ID ( hash )
            * Inits tensorboard writter 
            * Creates log directories 
                * tensorboard logs 
                * model checkpoints
                * plots
                * results
            * Creates agent configuration JSON file 
            * Creates environment results writer and it's JSON file 
              in order to track how the each agent with it's own configuration 
              is behaving in given environment
        """
        # Generate Agent UUID based on it's JSON configuration 
        config = self.config.copy()
        self.hash = md5(config) #hashlib.md5(json.dumps(config,sort_keys=True, indent=2).encode('utf-8')).hexdigest()

        # Create tensorboard writer and the directories
        self.writer, self.writer_log_directory = create_writer(self.config['env_name'],self.hash)   

        # Create Agent configuration JSON file
        with open(self.writer_log_directory+'/config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        # Create results writer and results JSON
        self.results_writer = ResultsWriter(self.config['env_name'],self.hash)

    def __init_loggers(self):
        self.learning_log = LearningLogger(self.learning_log_loss_keys)
   
    def __init_environment(self):
        """
        Extracts data from gym environment 
            * env
            * actions:
                * number of possible actions 
                * if they are discrete or continuous
                * upper and lower bounds ( in case of being continuous ) 
            * observations (state):
                * shape 
                * number of parameters 
        """
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
        """
        Keeps learning while loop alive:
            * if timesteps = -1 * forever until achieves a running average of success_threshold_lookback episodes with score above success_threshold
            * if timesteps > -1 * Until current timestep is greater then timesteps or achieves a running average of success_threshold_lookback episodes  ... like above LOL
        """
        if timesteps == -1:
            return True
        else: 
            return timesteps > timestep

    def did_finnish_learning(self,success_threshold,episode):
        """
        Validates if all learning conditions are met.
        """
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
        """
        Runs at the beginning of the learning phase, before it starts
            * Tracks training duration
            * Gets the previous max score if exists 
            * Sets success threshold metrics
            * Validates success threshold metrics
            * Starts loggers 
        """

        # Start metrics 

        # Track how long it took to train
        self.timestamp_learning_start = time.time()

        # Check if it's training for the first time or not
        # Set max score equals to last training best max score or a small number
        # @TODO
        self.is_first_train = self._load_best_score()

        # Is it in strict mode ( only stops if all episodes are positive - for trading env )
        self.success_strict = success_strict

        # Check if success threshold or valid timesteps are set so it can train 
        self.validate_learn(timesteps,success_threshold,reset)

        # How many episodes in a row it needs to have a greater running average in order beaten the environment 
        self.running_reward.success_threshold_lookback = success_threshold_lookback
        self.learning_log.success_threshold_lookback = success_threshold_lookback
        self.success_threshold_lookback = success_threshold_lookback

        # The value that the running average has to be above in order to beat the environment
        self.success_threshold = success_threshold if success_threshold else self.env.success_threshold
        
        return  self.success_threshold

    def on_learn_end(self,plot_results):
        """
        Runs at the end of the learning phase, after it ends
            * Tracks training duration
            * Closes environment 
            * Plots a graph with the results and stores it
        """

        self.timestamp_learning_end = time.time()
        self.training_duration = parse_time(self.timestamp_learning_end - self.timestamp_learning_start)

        print('* Learning cycle took {} to finnish.'.format(self.training_duration))

        self.env.close()

        if plot_results:
            self.plot_learning_results()

    def on_learn_episode_end(self,
        score,
        log_every,
        log_level,
        success_threshold,
        worker=False, 
        success_strict=False
    ):
        """
        Runs at the end of every episode
            * Updates running rewards
            * Logs results 
            * Tensorboards results 
            * If has achieved a maximum running average reward:
                * Checkpoints models
                * Stores max avg reward @ environment/results.json @ agent.hash  
        """
        self.running_reward.step(score)
    
        self.learning_log.episode(
            log_every,
            score,
            self.running_reward.moving_average, 
            log_level=log_level,
            worker=worker,
            success_strict=success_strict
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
        """
        Some agents need a epsilon to decrement on give ocasiation.
        """
        #self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > 0.01 else 0.01
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def plot_learning_results(self):
        """
        Plots a line plot with:
            * Blue line: Learning moving average score evolution
            * Red line: Smoothed earning moving average score evolution
        Saves the plot image in the plot results folder 
        """

        ysmoothed = gaussian_filter1d(self.running_reward.reward_history, sigma=10)

        plt.figure(figsize=(16,4))
        plt.plot(self.running_reward.reward_history, alpha=0.75, label="moving average reward history")
        plt.plot(np.zeros(len(self.running_reward.reward_history)), color="black", alpha=0.5, label="baseline")
        plt.plot(ysmoothed, color="red" , label="smoothed moving average history")
        plt.legend() 
        plt.title('Learning cycle score moving average evolution')
        # Or save before show
        fig = plt.gcf()
        plt.show()
        

        # Store
        file_id = get_number_of_files_in_folder(self.writer_log_directory+'/plots')
        file_name = 'training_results_'+str(file_id)+'.png'
        fig.savefig(self.writer_log_directory+'/plots/'+file_name)

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError
