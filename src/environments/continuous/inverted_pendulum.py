import gym 

ENV_NAME = 'Pendulum-v1'
SUCCESS_THRESHOLD = -150

def environment():

    print('''
| ---------------------------------
| {}
| 
| Action space: Discrete with low state-space
| Environment beated threshold: {}
| ----------------------------------------------------------   

'''.format(ENV_NAME,SUCCESS_THRESHOLD))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env.success_threshold = -150
    
    return env

