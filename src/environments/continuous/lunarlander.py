import gym 


ENV_NAME = 'LunarLanderContinuous-v2'
SUCCESS_THRESHOLD = 200

def environment(describe=True):
    if describe:
        print('''
    | ---------------------------------
    | {}
    | 
    | Action space: Continuous with low state-space
    | Environment beated threshold: {}
    | ----------------------------------------------------------   

    '''.format(ENV_NAME,SUCCESS_THRESHOLD))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env.success_threshold = SUCCESS_THRESHOLD
    
    return env
