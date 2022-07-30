from sre_constants import SUCCESS
import gym 

ENV_NAME = 'CartPole-v1'
SUCCESS_THRESHOLD = 200

def environment(describe=True):
    if describe:
        print('''
    | ---------------------------------
    | {}
    | Action space: Discrete with high state-space
    | Environment beated threshold: {}
    | Dev notes:
    |   * Agents that track State/Action combinations like 
    |     Q learning will fail due to high state space
    | ----------------------------------------------------------   

    '''.format(ENV_NAME,SUCCESS_THRESHOLD))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env.success_threshold = SUCCESS_THRESHOLD
    
    return env