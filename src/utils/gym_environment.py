import gym 


class GymEnvironment:
    def __init__(self,env):
        
        self.__init_environment(env)
        self.__extract_environment_properties()


    def __init_environment(self,env):
        # Might be a gym environment name or a function ( in case of my trading environment )
        if type(env) == str:
            self.env = gym.make(env)
        else:
            self.env = env()
    
    def __extract_environment_properties(self):
        # Gets num_actions 
        # Type of action space, and so on
        if type(self.env.action_space) == gym.spaces.Discrete:
            self.action_space_mode = "discrete"
            self.n_actions = self.env.action_space.n
            self.actions = list(range(self.n_actions))
            self.action_upper_bounds = False
            self.action_lower_bounds = False
        else:
            self.action_space_mode = 'continuous'
            self.n_actions = self.env.action_space.shape[0]
            self.actions = list(range(self.n_actions))
            # Refactor this. I might have other bounds
            self.action_upper_bounds = self.env.action_space.high
            self.action_lower_bounds = self.env.action_space.low

        if (len(self.env.observation_space.shape) == 1):
            self.n_inputs = self.env.observation_space.shape[0]
        else:
            self.n_inputs = self.env.observation_space.shape[0] * self.env.observation_space.shape[1] 


        
        self.observation_shape = self.env.observation_space.shape
