import gym
import pybullet_envs

ENV_NAME = 'InvertedPendulumBulletEnv-v0'
SUCCESS_THRESHOLD = 200

def environment(describe=True):
    if describe:
        print('''
    | ---------------------------------
    | {}
    | 
    | Action space: Continuous with high action-space
    | Environment beated threshold: {}
    | Dev notes:
    |   * Doesn't work with multiprocessing
    | ----------------------------------------------------------   

    '''.format(ENV_NAME,SUCCESS_THRESHOLD))
    
    # Will add a bit more of exploration so the agent can learn better
    env = gym.make(ENV_NAME)
    env.success_threshold = SUCCESS_THRESHOLD
    return env
