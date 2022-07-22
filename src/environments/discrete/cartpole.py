import gym 

ENV_NAME = 'CartPole-v1'

def environment():

    print('''
| ---------------------------------
| {}
| Action space:
|   * Discrete with high state-space
| Dev notes:
|   * Agents that track State/Action combinations like 
|     Q learning will fail due to high state space
| ----------------------------------------------------------   

'''.format(ENV_NAME))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env.success_threshold = 200
    
    return env