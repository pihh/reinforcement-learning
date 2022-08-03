import copy
import numpy as np

# gaes is better
# Compute the gamma-discounted rewards over an episode
# We apply the discount and normalize it to avoid big variability of rewards

def discounted_rewards(gamma=0.99):

    def fn(rewards):
        """
        Time discounted rewards, offers a way to measure the quality of a action 
        Measures the imediate reward but also the long term rewards
        ----------
        
        Args:
        rewards [float] : List of all imediate rewards  
        gamma   float   : Temporal discount factor
        """

        running_add = 0
        discounted_r = np.zeros_like(rewards)

        for i in reversed(range(0,len(rewards))):
            running_add = running_add * gamma + rewards[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        
        return discounted_r
    return fn



def generalized_advantage_estimation(gamma = 0.99, lamda = 0.9):
    """
    Generalized advantage estimation
    It's percieved as a better alternative than discounted rewards

    ----------
    
    Args:
    rewards     [float] : List of all imediate rewards  
    dones       [int]   : Episode end
    values      [object]: Episode state
    next_values [object]: Episode next state
    gamma       float   : Temporal discount factor
    lambda      float   : Smoothing parameter used for reducing the variance in training
    """
    def fn(rewards, dones, values, next_values, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        
        return np.vstack(gaes), np.vstack(target)

    return fn