import os
import json
import random
import hashlib

import numpy as np
import pandas as pd

from gym import Env
from gym.utils import seeding
from gym.spaces import Discrete, Box

from dotwiz import DotWiz
from collections import deque

from src.constants import FEES
from src.constants import ACTIONS
from src.utils.data import Downloader
from src.utils.data import FeatureEngeneer
from src.utils.trading_graph import TradingGraph

### ACTIONS
# -1.5 -> -0.5 -> sell
# -0.5 -> 0.5  -> hold
# 0.5  -> 1.5  -> buy

ACTION_AREA_SIZE = 1
ACTION_DEAD_AREA = [
    -1.5+ACTION_AREA_SIZE,
    1.5-ACTION_AREA_SIZE
] 

SPEC =  DotWiz()
SPEC.id = 'StockTradingEnvironment-v0'

class StockTradingEnvironment(Env):
    metadata = {'render.modes': ['human']}

    spec = SPEC

    def __init__(self,
                lookback=10, # 2 weeks
                window_size=66, # 3 months
                continuous=False,
                start_date= False,
                end_date=False,
                train_percentage=0.8,
                use_technical_indicators= [],
                use_sentiment_analysis=True,
                use_cboe_vix = True,
                use_trends = True,
                use_fear_and_greed=True,
                use_market_volatility= "DOW_30",
                ticker="AAPL",
                initial_investment=100000,
                # Punish it for doing nothing ? 
                inertness_punishment_method= None, # step , hold
                inertness_punishment_value = 0,#0.001,
                maximum_stocks_held=10, # For normalization purposes
                fees=FEES,
                seed=314,
                mode="train"
        ):

        self.lookback = lookback
        self.window_size = window_size
        self.continuous = continuous
        self.start_date = start_date
        self.end_date = end_date
        self.train_percentage = train_percentage
        self.technical_indicators = use_technical_indicators
        self.sentiment_analysis = use_sentiment_analysis
        self.use_cboe_vix = use_cboe_vix
        self.use_trends = use_trends
        self.use_fear_and_greed = use_fear_and_greed
        self.use_sentiment_analysis = use_sentiment_analysis
        self.use_market_volatility = use_market_volatility
        self.ticker = ticker.lower()
        self.initial_investment = initial_investment
        self.maximum_stocks_held = maximum_stocks_held
        self.fees = fees
        self.mode = mode


        self.__init_seed(seed)
        self.__init_dataset()
        self.__init_spaces()
        self.__init_buffers()
        self.__init_punishment(inertness_punishment_method,inertness_punishment_value)

    def __init_seed(self, seed=None):
        self._seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init_punishment(self,method,value):
        if method == None or value == 0:
            self.intertness_punishment = False
            def fn(action,reward):
                return reward
        elif method == "step":
            def fn(action,reward):
                return reward - value

        elif method == "hold":
            def fn(action,reward):
                if action > ACTION_DEAD_AREA[0] and action < ACTION_DEAD_AREA[1]:
                    return reward - value
                else:
                    return reward

        self.punishment = fn

    def __init_dataset(self):
        # Get dataset name by it's caracteristics
        name_params = {
            "ticker":self.ticker,
            "lookback":str(self.lookback) ,
            "window_size":str(self.window_size),
            "start_date":str(self.start_date),
            "end_date": str(self.end_date),
            "use_technical_indicators" : " ".join(self.technical_indicators),
            "use_sentiment_analysis": str(self.use_sentiment_analysis),
            "use_cboe_vix": str(self.use_cboe_vix),
            "use_trends": str(self.use_trends),
            "use_fear_and_greed": str(self.use_fear_and_greed),
            "use_market_volatility": str(self.use_market_volatility),
            "ticker": self.ticker
        }
        #raw_name = json.dumps(name_params)
        df_name = hashlib.md5(json.dumps(name_params,sort_keys=True, indent=2).encode('utf-8')).hexdigest()
        df_path = 'src/environments/continuous/stock_trading/datasets/'+df_name

        self.df_name = df_name
        self.df_path = df_path

        if not os.path.exists(df_path):
            # Se nao existir uma pasta com este nome
            # Cria pasta com este nome
            # Normaliza o dataframe
            # Guarda raw e normalizado
            # Parte raw e normalizado em slices de lookback + window_size ( para ter vários cenários possiveis )
            # Criar um config file com a configuração do ambiente e guarda-a nessa pasta

            os.mkdir(df_path)
            downloader = Downloader()
            df = downloader.fetch_data()

            # Add features to datasets
            feature_engeneer = FeatureEngeneer(df)
            if self.use_cboe_vix:
                df = feature_engeneer.vix(df)
                print(1,len(df))
            if self.use_market_volatility:
                df = feature_engeneer.turbulence(df)

            if len(self.technical_indicators)> 0:
                df = feature_engeneer.technical_indicators(df, self.technical_indicators)

            if self.use_sentiment_analysis:
                df = feature_engeneer.sentiment_analysis(df,self.ticker,key="title")
                df = feature_engeneer.sentiment_analysis(df,self.ticker,key="body")

            if self.use_trends:
                df = feature_engeneer.trends(df,self.ticker)

            if self.use_fear_and_greed:
                df = feature_engeneer.fear_and_greed(df)


            df = feature_engeneer.cleanup(df)
            df.reset_index(drop=True,inplace=True)
            df.to_csv(df_path+'/raw_dataframe.csv')

            # Normalize dataframe
            df_norm = df.copy()
            for col in df_norm.columns:
                if col.startswith('macd'):
                    df_norm[col] = df_norm[col]/ df_norm['close']

                if col.startswith('macd'):
                    df_norm[col] = df_norm[col]/ df_norm['close']
                if col.startswith('boll_'):
                    df_norm[col] = df_norm[col]/ df_norm['close']
                if col.startswith('dx_'):
                    df_norm[col] = df_norm[col]/ 100
                if col.startswith('rsi_'):
                    df_norm[col] = df_norm[col]/ 100
                if col.startswith('cci_'):
                    df_norm[col] = df_norm[col]/ 250
                if col.endswith('_sma'):
                    df_norm[col] = df_norm[col]/ df_norm['close']

                if col== 'day':
                    df_norm[col] = df_norm[col]/ 4

                if col in ['open','close','high','low','volume']:
                    df_norm[col] = df_norm[col].pct_change()

            df_norm.fillna(0,inplace=True)
            df_norm.reset_index(drop=True,inplace=True)
            df_norm.to_csv(df_path+'/norm_dataframe.csv')

            # Slice dataframes
            n_dataframes = 0
            n_columns = len(df.columns) -2 # Remove date and ticker

            for i in range(self.lookback, len(df)- self.window_size):
                raw_slice = df.iloc[i-self.lookback:i+self.window_size]
                raw_slice.reset_index(drop=True,inplace=True)
                raw_slice.to_csv(df_path+'/raw_slice_'+str(i-self.lookback)+'.csv')

                norm_slice = df_norm.iloc[i-self.lookback:i+self.window_size]
                norm_slice.reset_index(drop=True,inplace=True)
                norm_slice.to_csv(df_path+'/norm_slice_'+str(i-self.lookback)+'.csv')
                n_dataframes +=1

            with open(df_path+'/config.json', 'w') as f:
                json.dump(name_params, f, indent=2)

            pd.DataFrame([[n_dataframes,n_columns]],columns=['n_dataframes','n_columns']).to_csv(df_path+'/config.csv')

       
        config = pd.read_csv(df_path+'/config.csv')
        df = self.load_dataset_by_index(0)

        n_dataframes = config.iloc[0].n_dataframes
        n_columns = config.iloc[0].n_columns
        
        self.n_dataframes = n_dataframes
        self.n_columns = n_columns
        
        self.train_dataframe_id_range = [0,int(self.train_percentage * n_dataframes)-1]
        self.test_dataframe_id_range = [int(self.train_percentage * n_dataframes),n_dataframes]

        # Verifica quantos ficheiros raw tem a pasta
        # Faz range de 0 a int(train_percentage) - self.train_df_range e that's it, é excusado carregar agora o dataset, carrega no reset

    def __init_spaces(self):
        # Quanto já investiu, quanto retorno tem de momento, retorno total
        self.portfolio_history_params = [
            'portfolio_value', # How is the agent performing -> iniciou com 100, está com 110 -> 10% increase
            'cash_in_hand', # Começou com 100 , tem 50 -> 50%
            'n_stocks_held', # Max stocks held to normalize?
            # Comprou 1@100 + 1@50 => avg price é 75 
            # Se average stock price avg é 75 e current price é 100 -> está 100*100/75 => 133% => 33% mais caro do que comprei 
            'stock_price_avg_compared', 
        ]
        self.orders_history_params = [
            'held',
            'sold',
            'bought'] # bool % %

        # Tracks how much the trader has invested and how it's performing
        # How much of the cash available to invest is invested ( initial_balance is the most he invests )
        # How much this investment is paying off ( invested 100 in stock and has 90: -0.1 )
        # How much value has he created ? Started with 1000 of initial balance and now I have 1010 - so 0.01
        #self.portfolio_history_params = ['invested','current_stock_investment_return', 'net_worth_return'] # % % %

        # Spaces
        # lookback = n days ( for lstm and cnn ) -> 3,20 -> 3 days with 20 indicators
        input_size = self.n_columns + len(self.orders_history_params) + len(self.portfolio_history_params)
        
        shape = (self.lookback, input_size)
        
        self.action_space = Box(low = -1.5, high = 1.5,shape = (1,))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)
        self.input_shape = shape

    def __init_buffers(self):
        self.history_params_orders = []
        self.history_params_portfolio = []
        self.history_params_market = []
        self.history_params_news = []
        self.history_params_indicators = []

        self.reset_history()
        # self.orders_history.append([0,0,0])
        #
        # # Portfolio
        # self.portfolio_history.append([0,0,0])  # % % %
        #
        # # Market history tracks OHLC
        # self.market_history.append(self.df_normalized.loc[current_step][self.market_history_params])
        #
        # # News history tracks news
        # self.news_history.append(self.df_normalized.loc[current_step][self.news_history_params])
        #
        # # Indicators history tracks indicators
        # self.indicators_history.append(self.df_normalized.loc[current_step][self.indicators_history_params])

    def __init_visualization(self):
        self.visualization = TradingGraph(render_range=66, show_reward=True, show_indicators=True) # init visualization

    def _action(self,_action):
        """ Compra ou vende e quanto % do dinheiro inicial deveria investir ? """
        # bounds = ACTION_DEAD_AREA
        # bound_normalization = (1 - bounds[1])


        # if _action < bounds[0]:
        #     # Vende
        #     action = ACTIONS.SELL

        #     # percentage que o algoritmo decidiu vender
        #     amount = -1 * ((_action - bounds[0]) / bound_normalization) #(_action/bounds[0])

        #     # reinicia o castigo
        #     self.punish_value = 0

        # elif _action > bounds[1]:
        #     # Compra
        #     action = ACTIONS.BUY

        #     # percentage que o algoritmo decidiu
        #     amount_ordered = ((_action - bounds[1]) / bound_normalization)

        #     # máximo possível
        #     amount = min(amount_ordered,1-self.invested_percentage)

        #     # reinicia o castigo
        #     self.punish_value = 0
        # else:
        #     action = ACTIONS.HOLD
        #     amount = 0

        # return action,amount
        return ACTIONS.HOLD, 0

    def _trade(self,_action):
        # Track what we have done in this episode
        # self.stock_bought = 0
        # self.stock_sold = 0
        # self.episode_penalty = False

        # #self.punish_value += self.inertness_punishment

        # current_price = self._get_current_price()

        # action,amount = self._action(_action)

        # if action == ACTIONS.HOLD:
        #     self.episode_history.append(["hold",action,current_price,self.episode_step])
        # else:

        #     if action ==ACTIONS.BUY:
        #         # Preço com os descontos das fees
        #         discounted_price = current_price*(1+self.fees[0])

        #         available_balance = self.balance
        #         available_balance_percentage = self.balance / self.initial_investment
        #         available_investment_percentage = min(available_balance_percentage,amount)

        #         traded_cash = (available_investment_percentage * self.initial_investment)
        #         traded_stocks = traded_cash / discounted_price

        #         self.balance -= traded_cash
        #         self.stock_held += traded_stocks
        #         self.stock_bought = available_investment_percentage

        #         self.invested_percentage += available_investment_percentage

        #         if traded_stocks > 0:
        #             # Calculate price means
        #             self.stock_prices.append(current_price)
        #             self.stock_volumes.append(traded_stocks)

        #             price_mean_upper_bound = 0
        #             price_mean_lower_bound = 0

        #             for i in range(len(self.stock_prices)):
        #                 p = self.stock_prices[i]
        #                 a = self.stock_volumes[i]
        #                 price_mean_upper_bound += p * a
        #                 price_mean_lower_bound += a

        #             self.stock_price_mean = price_mean_upper_bound/price_mean_lower_bound

        #         self.episode_history.append(["buy",action,discounted_price,self.episode_step])
        #         self.episode_orders += 1

        #         if self.track_orders :
        #             date = self.dates.loc[self.current_step, 'date'] # for visualization
        #             open = self.df.loc[self.current_step, 'open']
        #             close = self.df.loc[self.current_step, 'close']
        #             high = self.df.loc[self.current_step, 'high'] # for visualization
        #             low = self.df.loc[self.current_step, 'low'] # for visualization

        #             self.trades.append({
        #                 "date":date,
        #                 "open":open,
        #                 "close":close,
        #                 'high':high,
        #                 'low':low,
        #                 'total': traded_stocks,
        #                 'type': "buy",
        #                 "current_price":discounted_price,
        #                 "step": self.episode_step
        #             })
        #     else:
        #         if self.stock_held > 0:
        #             discounted_price = current_price*(1+self.fees[1])
        #             traded_stocks = self.stock_held * amount
        #             traded_cash = discounted_price * traded_stocks

        #             self.balance += traded_cash
        #             self.stock_held -= traded_stocks
        #             self.stock_sold = traded_cash/self.initial_investment

        #             self.invested_percentage = max(0, self.invested_percentage - (traded_cash/self.initial_investment))

        #             self.episode_history.append(["sell",action,discounted_price,self.episode_step])

        #             self.episode_orders += 1

        #             if self.track_orders :
        #                 date = self.dates.loc[self.current_step, 'date'] # for visualization
        #                 open = self.df.loc[self.current_step, 'open']
        #                 close = self.df.loc[self.current_step, 'close']
        #                 high = self.df.loc[self.current_step, 'high'] # for visualization
        #                 low = self.df.loc[self.current_step, 'low'] # for visualization

        #                 self.trades.append({
        #                     "date":date,
        #                     "open":open,
        #                     "close":close,
        #                     'high':high,
        #                     'low':low,
        #                     'total': traded_stocks,
        #                     'type': "buy",
        #                     "current_price":discounted_price,
        #                     "step": self.episode_step
        #                 })
        pass

    def _get_current_price(self):
        # if not self.volatil_price_mode:
        #     return self.df.loc[self.current_step, 'close']
        # else:
        #     return random.uniform(
        #          self.df.loc[self.current_step, 'low'],
        #          self.df.loc[self.current_step, 'high'])
        return self.df.iloc[self.current_step, 'close']

    def _normalize_portfolio(self,i):

        # portfolio_value%  cash_held% stocks_held_% stock_price_avg_comp%

        PARAM = 'close'
    #     # if self.pessimistic_mode:
    #     #     PARAM = 'high'

        portfolio_value = self.portfolio_value/self.initial_investment
        cash_in_hand = self.cash_in_hand/self.initial_investment
        n_stock_held = self.n_stock_held/ self.maximum_stocks_held
        stock_price_avg_comp = 1 if len(self.stock_prices) == 0 else np.mean(self.stock_prices)/self.df.iloc[i][PARAM] -1# No stocks so the value of a stock compared to what I have is itself


        return [portfolio_value,cash_in_hand,n_stock_held,stock_price_avg_comp]  # % % %

    def _state(self):
        #return np.concatenate((self.orders_history,self.portfolio_history, self.market_history,self.news_history,self.indicators_history), axis=1)
        state = np.concatenate((
            self.orders_history,
            self.portfolio_history,
            self.market_history)
            ,axis=1)

        self.state = state 

    def _next_state(self):
        i = self.current_step

        held = 1
        if self.stock_sold > 0 or self.stock_bought > 0:
            held = 0

        # # Add order tracking
        self.orders_history.append([i,held,self.stock_sold,self.stock_bought])

        # # Add portfoluio state tracking
        self.portfolio_history.append(self._normalize_portfolio(i))  # % % %

        # # Market history tracks OHLC
        self.market_history.append(self.df_norm.iloc[i])

        # # News history tracks news
        # self.news_history.append(self.df_normalized.loc[i][self.news_history_params])

        # # Indicators history tracks indicators
        # self.indicators_history.append(self.df_normalized.loc[i][self.indicators_history_params])

        # # Return state
        # self.state = self._state()
        self._state()
        # return self.state


    def _calculate_reward(self):
        # PARAM = 'close'
        # # if self.pessimistic_mode:
        # #     PARAM = 'low'

        # # Deveria ser péssimista e usar low?
        # Price = self.df.loc[self.current_step, PARAM]

        # # Calculate net worth difference
        # self.prev_net_worth = self.net_worth
        # self.net_worth = self.balance + self.stock_held * Price #*(1-self.fees[1])

        # reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth - self.punish_value

        # return reward
        pass

    def reset_history(self):
        self.history_orders = deque(maxlen=self.lookback)
        self.history_portfolio = deque(maxlen=self.lookback)
        self.history_market = deque(maxlen=self.lookback)
        self.history_news = deque(maxlen=self.lookback)
        self.history_indicators = deque(maxlen=self.lookback)
        self.history_trades = deque(maxlen=self.lookback)
        
    def load_dataset_by_index(self,idx):
        self.df = pd.read_csv(self.df_path+'/raw_slice_'+str(idx)+'.csv')
        self.df_norm = pd.read_csv(self.df_path+'/norm_slice_'+str(idx)+'.csv')
    
        for df in [self.df,self.df_norm]:
            try:
                df.drop(columns=['Unnamed: 0'],inplace=True)
            except:
                pass

            try:
                df.set_index('date',inplace=True)
            except:
                pass 
        
            df.drop(columns=['ticker'],inplace=True)

    def step(self,action):
        if type(action)== np.ndarray :
            action = action[0]

        # Executa ação
        #self._trade(action)

        # Verifica a recompensa
        #reward = self._calculate_reward()
        reward = 0

        # Check if we done - when ended episode or cannot invest anymore
        if self.current_step == self.window_size -1:
            done = True
            # Calculate sharp ?
            # Add penalization in the end of the episode?
        else:
            done = False

        # Get the next state
        self._next_state()
        
        print(self.current_step)
        # Update stepper
        self.episode_step += 1
        self.current_step +=1

        # Return gym env step structure
        return self.state, reward, done , {}

    def render(self):
        img = self.visualization.render(self.df.iloc[self.current_step], self.net_worth, self.trades)
        return img

    def reset(self):

        # Get a random dataset 
        if self.mode == "train":
            dataset_idx_start=self.train_dataframe_id_range[0]
            dataset_idx_end=self.train_dataframe_id_range[1]
        else:
            dataset_idx_start=self.test_dataframe_id_range[0]
            dataset_idx_end=self.test_dataframe_id_range[1]

        self.dataset_idx = np.random.randint(dataset_idx_start, high=dataset_idx_end, size=1, dtype=int)[0]
        
        self.load_dataset_by_index(self.dataset_idx)

        # # State queues
        self.orders_history = deque(maxlen=self.lookback)    # Order tracking during episode
        self.portfolio_history = deque(maxlen=self.lookback) # Portfolio evolution tracking during episode
        self.market_history = deque(maxlen=self.lookback)    # Dataframe data during episode

        # History queues
        self.trades = deque(maxlen=self.window_size)

        # Trader state
        self.portfolio_value = self.initial_investment
        self.cash_in_hand = self.initial_investment
        self.n_stock_held = 0
        self.stock_price_avg_comp = 1 # No stocks so the value of a stock compared to what I have is itself

        #self.balance = self.initial_investment
        #self.invested_percentage = 0
        self.net_worth = self.initial_investment
        self.prev_net_worth = self.initial_investment

        self.stock_held = 0
        self.stock_sold = 0
        self.stock_bought = 0

        # Track the prices I've bought to check how the current is performing
        self.stock_prices = []
        self.stock_volumes = []
        self.stock_price_mean = 0

        # # Punishment
        # self.punish_value = 0

        # # For reward calculation
        # self.episode_orders = 0
        # self.prev_episode_orders = 0
        # self.episode_history = []
        # self.episode_penalty = False

        # # Stepper
        self.episode_step = 0
        self.current_step = self.lookback

        # # Create the initial state
        #for i in reversed(range(self.lookback)):
        for i in range(self.lookback):
            current_step = self.current_step -i

            # Orders history tracks recent trader activity - held bought sold
            self.orders_history.append([i,0,0,0]) # Held, Sold, Bought

            # Portfolio
            self.portfolio_history.append([1,1,0,1])  # portfolio_value_%  cash_held_% stocks_held_% stock_price_avg_comp_%

            # Market history 
            self.market_history.append(self.df_norm.iloc[current_step])



        # # Create a new state
        # # Should I make a state object that tracks this shit?
        self._state()

        # return self.state
        return self.state 

    def learn(self,timesteps=-1, visualize=False):
        if visualize:
            self.__init_visualization()

        