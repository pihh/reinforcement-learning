from gym.envs.registration import register
from src.constants import FEES


register(
    id='StockTradingEnvironment-v0',
    entry_point='src.environments.continuous.stock_trading.v0:StockTradingEnvironment',
    kwargs={
        "lookback":10,
        "window_size":66,
        "use_technical_indicators": [],
        "use_sentiment_analysis":True,
        "use_cboe_vix": True,
        "use_trends": True,
        "use_fear_and_greed":True,
        "use_market_volatility": "DOW_30",
        "ticker":"AAPL",
        "initial_investment":100000,
        "inertness_punishment":0.001,
        "fees":FEES,
        "seed":314,
    }
)
