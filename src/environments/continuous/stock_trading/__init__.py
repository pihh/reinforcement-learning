from gym.envs.registration import register
from src.constants import FEES


register(
    id='StockTradingEnvironment-v0',
    entry_point='src.environments.continuous.stock_trading.v0:StockTradingEnvironment',
    kwargs={
        "lookback":21, # 1 month
        "window_size":126, # 6 month
        "continuous":False,
        "use_technical_indicators": [],
        "use_sentiment_analysis":True,
        "use_cboe_vix": True,
        "use_trends": True,
        "use_fear_and_greed":True,
        "use_market_volatility": "DOW_30",
        "ticker":"AAPL",
        "initial_investment":False,
        "inertness_punishment_method": None, # step , hold
        "inertness_punishment_value": 0,#0.001,
        "maximum_stocks_held":10,
        "fees":FEES,
        "seed":314,
        "mode":"train"
    }
)
