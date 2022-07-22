import gym 

ENV_NAME = 'MountainCarContinuous-v0'

def environment():

    print('''
| ---------------------------------
| {}
| Action space:
|   * Continuous with low state-space
| Dev notes:
|   * Switched _max_episode_steps from 200 to 1000 so 
|     the agent can explore better.
| ----------------------------------------------------------   

'''.format(ENV_NAME))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 1000
    env.success_threshold = -150
    
    return env