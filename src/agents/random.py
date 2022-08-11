import multiprocessing

from tqdm import tqdm

class RandomAgent:
  """Random Agent that will play the specified game

    Arguments:
      env_name: Name of the environment to be played
      max_eps: Maximum number of episodes to run agent for.
  """
  def __init__(self, environment, n_episodes):
      self.env = environment
      self.n_episodes = n_episodes
      # self.global_moving_average_reward = 0
      # self.res_queue = multiprocessing.Queue()

  def run(self):

    rewards = []
    for episode in tqdm(range(self.n_episodes)):
      done = False
      self.env.reset(visualize=False,mode="test")
      score = 0
      steps = 0
      while not done:
        # Sample randomly from the action space and step
        _, reward, done, _ = self.env.step(self.env.action_space.sample())
        steps += 1
        score += reward


      rewards.append(score)
    
    return rewards
   

    