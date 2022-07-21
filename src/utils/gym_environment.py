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
            self.action_space_mode = "Discrete"
            self.n_actions = self.env.action_space.n
            self.actions = range(self.env.action_space.n)
        else:
            self.action_space_mode = 'Continuous'
            self.n_actions = self.env.action_space.shape[0]
            self.actions = range(self.n_actions)
        
        self.observation_shape = self.env.observation_space.shape
