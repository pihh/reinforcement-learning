# from turtle import back
# import shutup
# shutup.please()

# from src.environments.continuous.pendulum import environment
# #from src.environments.discrete.cartpole import environment
# from src.agents.ppo import PpoAgent

# def env(describe=True):
#     e = environment(describe=describe)
#     e.success_threshold = 50
#     return e

# if __name__ == "__main__":
#     agent = PpoAgent(
#         env,
#         n_workers=2,
#         epochs=10,
#         batch_size=2000 
#     )
#     agent.learn(log_every=10)


import shutup
shutup.please()

import gym
import src.environments.continuous.stock_trading  

from src.agents.ppo import PpoAgent

if __name__ == '__main__':

    def environment(describe=True):
        env = gym.make('StockTradingEnvironment-v0',
            use_technical_indicators= [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ])
    
        env.success_threshold =0.25 # 25%

        return env

    agent=PpoAgent(
        environment,
        actor_learning_rate=0.000025,
        critic_learning_rate=0.000025,
        policy="CNN",
        n_workers=2)
    agent.learn(
        timesteps=-1, 
        log_every=50,
        success_threshold_lookback=1000,
        success_strict=True,
    )