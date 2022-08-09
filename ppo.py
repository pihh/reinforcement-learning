from turtle import back
import shutup
shutup.please()

from src.environments.continuous.pendulum import environment
from src.agents.ppo import PpoAgent

def env(describe=True):
    e = environment(describe=describe)
    e.success_threshold = 50
    return e

if __name__ == "__main__":
    agent = PpoAgent(
        env,
        n_workers=2,
        epochs=10,
        batch_size=2000 
    )
    agent.learn(log_every=10)
