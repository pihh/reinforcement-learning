import gym
import src.environments.continuous.stock_trading

from src.constants import FEES

ENV_NAME = "StockTradingEnvironment-v0"


def environment(
    describe=True,
    lookback=21,
    window_size=126,
    continuous=False,
    start_date=False,
    end_date=False,
    train_percentage=0.8,
    use_technical_indicators=[
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ],
    use_sentiment_analysis=True,
    use_cboe_vix=True,
    use_trends=True,
    use_fear_and_greed=True,
    use_market_volatility="DOW_30",
    ticker="AAPL",
    initial_investment=None,
    inertness_punishment_method=None,
    inertness_punishment_value=0,
    maximum_stocks_held=10,
    fees=FEES,
    seed=314,
    mode="train",
):

    env = gym.make(
        ENV_NAME,
        lookback=lookback,
        window_size=window_size,
        continuous=continuous,
        start_date=start_date,
        end_date=end_date,
        train_percentage=train_percentage,
        use_technical_indicators=use_technical_indicators,
        use_sentiment_analysis=use_sentiment_analysis,
        use_cboe_vix=use_cboe_vix,
        use_trends=use_trends,
        use_fear_and_greed=use_fear_and_greed,
        use_market_volatility=use_market_volatility,
        ticker=ticker,
        initial_investment=initial_investment,
        inertness_punishment_method=inertness_punishment_method,
        inertness_punishment_value=inertness_punishment_value,
        maximum_stocks_held=maximum_stocks_held,
        fees=fees,
        seed=seed,
        mode=mode,
    )
    if describe:
        print(
            """
        | ---------------------------------
        | {}
        | 
        | Author: Pihh - pihh.rocks@gmail.com
        | Description: Configurable stock trading environment for my trading bot
        | Action space: Discrete with high state-space
        | Environment beated threshold: {}
        | ----------------------------------------------------------   

        """.format(
                env.spec.name, env.success_threshold
            )
        )

    # Will add a bit more of exploration so the agent can learn better

    return env
