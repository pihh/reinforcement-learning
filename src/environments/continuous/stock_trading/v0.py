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
                initial_investment=None,
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
        self.auto_investment =  initial_investment == False
        self.maximum_stocks_held = maximum_stocks_held
        self.fees = fees
        self.mode = mode


        self.__init_seed(seed)
        self.__init_dataset()
        self.__init_targets()
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
            df.to_csv(df_path+'/raw_dataframe.csv',index=False)

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
            df_norm.to_csv(df_path+'/norm_dataframe.csv',index=False)

            # Slice dataframes
            n_dataframes = 0
            n_columns = len(df.columns) -2 # Remove date and ticker

            # Track episode targets
            episode_targets = []
  
            # Verifica quantos ficheiros raw tem a pasta
            # Faz range de 0 a int(train_percentage) - self.train_df_range e that's it, é excusado carregar agora o dataset, carrega no reset

            for i in range(self.lookback, len(df)- self.window_size):
                raw_slice = df.iloc[i-self.lookback:i+self.window_size]
                raw_slice.reset_index(drop=True,inplace=True)
                raw_slice.to_csv(df_path+'/raw_slice_'+str(i-self.lookback)+'.csv',index=False)

                norm_slice = df_norm.iloc[i-self.lookback:i+self.window_size]
                norm_slice.reset_index(drop=True,inplace=True)
                norm_slice.to_csv(df_path+'/norm_slice_'+str(i-self.lookback)+'.csv',index=False)

                episode_targets.append([
                    self.get_episode_target(raw_slice),
                    raw_slice.iloc[self.lookback-1].open,
                    raw_slice.iloc[self.lookback-1].low,
                    raw_slice.iloc[self.lookback-1].high,
                    raw_slice.iloc[self.lookback-1].close,
                    raw_slice.iloc[self.lookback-1].volume,
                ])


                n_dataframes +=1


            # Define os targets de sucesso para cada dataframe

            with open(df_path+'/config.json', 'w') as f:
                json.dump(name_params, f, indent=2)

            pd.DataFrame([[n_dataframes,n_columns]],columns=['n_dataframes','n_columns']).to_csv(df_path+'/config.csv', index=False)
            pd.DataFrame(episode_targets,columns=['targets','open','low','high','close','volume']).to_csv(df_path+'/targets.csv', index=False)
       
        config = pd.read_csv(df_path+'/config.csv')
        episode_targets = pd.read_csv(df_path+'/targets.csv')

        n_dataframes = config.iloc[0].n_dataframes
        n_columns = config.iloc[0].n_columns
        
        self.n_dataframes = n_dataframes
        self.n_columns = n_columns
        
        self.train_dataframe_id_range = [0,int(self.train_percentage * n_dataframes)-1]
        self.test_dataframe_id_range = [int(self.train_percentage * n_dataframes),n_dataframes]

        # self.episode_targets = []
        # self.initial_investments = []
        # for i in range(self.train_dataframe_id_range[1]+1):
        #     if self.auto_investment:
        #         initial_investment = episode_targets.high.iloc[i] * self.maximum_stocks_held 
        #     else: 
        #         initial_investment = self.initial_investment #self._initial_investment_calculation(i)

        #     target = episode_targets.targets.iloc[i]
        #     self.episode_targets.append(initial_investment+target)
        #     self.initial_investments.append(initial_investment)

        # # Calculate the success threshold
        # success_threshold_targets = np.mean(self.episode_targets) #+ np.std(self.episode_targets)
        # success_threshold_investments = np.mean(self.initial_investments) #+ np.std(self.initial_investments)
        
        # self.success_threshold = (success_threshold_targets -success_threshold_investments)/ success_threshold_investments

        print()

    def __init_targets(self):

        episode_targets = pd.read_csv(self.df_path+'/targets.csv')

        self.episode_targets = []
        self.initial_investments = []

        for i in range(self.train_dataframe_id_range[1]+1):
            if self.auto_investment:
                initial_investment = episode_targets.high.iloc[i] * self.maximum_stocks_held 
            else: 
                initial_investment = self.initial_investment #self._initial_investment_calculation(i)

            target = episode_targets.targets.iloc[i]
            self.episode_targets.append(initial_investment+target)
            self.initial_investments.append(initial_investment)

        # Calculate the success threshold
        success_threshold_targets = np.mean(self.episode_targets) #+ np.std(self.episode_targets)
        success_threshold_investments = np.mean(self.initial_investments) #+ np.std(self.initial_investments)
        
        self.success_threshold = (success_threshold_targets -success_threshold_investments)/ success_threshold_investments

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
        
        if self.continuous:
            self.action_space = Box(low = -1.5, high = 1.5,shape = (1,))
            self.ACTIONS = ACTIONS
        else:
            self.action_space = Discrete(len(ACTIONS))

            ACTIONS.SELL = 0
            ACTIONS.HOLD = 1
            ACTIONS.BUY = 2
            self.ACTIONS = ACTIONS
        
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)
        self.input_shape = shape

    def __init_buffers(self):
        # @TODO
        self.history_params_orders = []
        self.history_params_portfolio = []
        self.history_params_market = []
        self.history_params_news = []
        self.history_params_indicators = []

        self.reset_history()

    def _action(self,actions):
        """ Compra ou vende e quanto % do dinheiro inicial deveria investir ? """
        if self.continuous:
            raise Exception('TODO CRL')
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
            #return ACTIONS.HOLD, 0
        else:
            #print(actions, self.ACTIONS)
            action = actions

            if actions == self.ACTIONS.SELL:
                # print('sell')
                # Sell all 
                amount = self.stock_held
            elif actions == self.ACTIONS.BUY: 
                # print('buy')
                amount = 1 
            else:
                # print('hold')
                amount = 0

        return action,amount

    def _trade(self,action,amount):
        # Track what we have done in this episode
        self.stock_bought = 0
        self.stock_sold = 0

        current_price = self._get_current_price()
        discounted_price = current_price
        position = 'hold'

        if action == self.ACTIONS.BUY:
            position = 'buy'
            discounted_price =self._get_current_buying_price()

            # Put it on history
            self.stock_bought = 1

            # Verifica se posso comprar
            if self.stock_held < self.maximum_stocks_held and self.cash_in_hand >= discounted_price *amount:

                # Actualiza contador de ações
                self.stock_held += amount

                # Actualiza contador de preços de açoes
                self.stock_prices.append(current_price)
                self.stock_volumes.append(amount)
                self.stock_price_mean = np.mean(np.array(self.stock_prices) * np.array(self.stock_volumes))

                ## Actualiza cash in hand 
                self.cash_in_hand -= discounted_price 

                # Segue pedidos 
                # @TODO
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
            else:
                amount = 0

        elif action == self.ACTIONS.SELL:
            # Sells all in a row
            position = 'sell'
            discounted_price = self._get_current_selling_price()
            
            # Put it on history
            self.stock_sold = 1

            # Verifica se posso vender
            if self.stock_held > 0:
                if self.continuous:
                    raise Exception('@TODO sell continuous')
                else:
                    # Actualiza cash in hand
                    sale_profit = discounted_price * self.stock_held
                    self.cash_in_hand += sale_profit
                    
                    # Actualiza contador de ações e preços
                    self.stock_held = 0
                    self.stock_prices.clear()
                    self.stock_volumes.clear()
                    self.stock_price_mean = 0
            else:
                amount = 0

        self.episode_history.append([self.episode_step, position, action, current_price])

    def _normalize_portfolio(self,i):

        # portfolio_value%  cash_held% stocks_held_% stock_price_avg_comp%

        PARAM = 'close'
    #     # if self.pessimistic_mode:
    #     #     PARAM = 'high'

        portfolio_value = self.portfolio_value/self.initial_investment
        cash_in_hand = self.cash_in_hand/self.initial_investment
        stock_held = self.stock_held/ self.maximum_stocks_held
        stock_price_avg_comp = 1 if len(self.stock_prices) == 0 else self.stock_price_mean/self._get_current_price() -1 # No stocks so the value of a stock compared to what I have is itself

        return [portfolio_value,cash_in_hand,stock_held,stock_price_avg_comp]  # % % %

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
        self.orders_history.append([held,self.stock_sold,self.stock_bought])

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
        # return self.state
        self._state()

    def _get_current_buying_price(self):
        # Compra a um preço e adiciona comissão
        return self._get_current_price() * (1+self.fees.BUY) 

    def _get_current_selling_price(self):
        # Compra a um preço e paha comissão
        return self._get_current_price() * (1-self.fees.SELL) 

    def _get_current_price(self, optimistic=True):
        return self.df.iloc[self.current_step -1]['close']

    def _calculate_reward(self):
        # Using portfolio evolution
        # Could use portfolio max possible evolution
        # Could also use portfolio distance from target

        # PARAM = 'close'
        # # if self.pessimistic_mode:
        # #     PARAM = 'low'

        # # Calculate net worth difference
        #self.prev_portfolio_value = self.portfolio_value -> set at begining of episode before take action
        self.portfolio_value = self.cash_in_hand + self.stock_held * self._get_current_selling_price()
        reward = (self.portfolio_value - self.prev_portfolio_value) / self.initial_investment

        return reward

    def _reached_end_of_episode(self):
        return self.episode_step == self.window_size -1

    def _beated_environment(self,action):
        if action == self.ACTIONS.SELL and self.portfolio_value >= self.episode_target:
            self.environment_beaten = True
            return True 
        else:
            return False

    def _calculate_done(self,action):
        # Reached the end of the episode 
        if self._reached_end_of_episode() or self._beated_environment(action):
            return True 
        else:
            return False 

    def _initial_investment_calculation(self,idx):
        if self.auto_investment:
            return self.initial_investments[idx]
        else:
            return self.initial_investment
            
    def _set_initial_investment(self, idx):
        self.initial_investment = self._initial_investment_calculation(idx)

    def _set_episode_target(self,idx):
        self.episode_target = self.episode_targets[idx]

    def get_episode_target(self,df):
        # For now the target is the maximum profit in a single trade - fees 
        prices = df.close.values
        prices_sell = prices * (1-self.fees.SELL)
        prices_buy = prices * (1+self.fees.BUY)

        mins = [] 
        maxs = [] 

        for i in range(len(prices)):
            mins.append(min(prices_buy[0:i+1]))
            maxs.append(max(prices_sell[i:]))
            
        # mins,maxs
        global_max_single_trade_profit = 0
        for i in range(len(prices)):
            _min = mins[i]
            _max = maxs[i]           

            local_max_single_trade_profit = _max-_min
            if local_max_single_trade_profit > global_max_single_trade_profit:
                global_max_single_trade_profit = local_max_single_trade_profit

        return global_max_single_trade_profit

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
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date',inplace=True)
            except:
                pass 
        
            df.drop(columns=['ticker'],inplace=True)

    def step(self,action):

        self.prev_portfolio_value = self.portfolio_value
        self.stock_sold = 0
        self.stock_bought = 0

        action, amount = self._action(action)

        # Executa ação
        self._trade(action,amount)

        # Verifica a recompensa
        reward = self._calculate_reward()

        # Check if we done - when ended episode or cannot invest anymore
        done = self._calculate_done(action)

        if done and self.environment_beaten:
            # Should add some incentive for beating before time no?
            pass 

        #if done and self.current_step < self.window_size -1:

        # if self.current_step == self.window_size -1:
        #     done = True
        #     # Calculate sharp ?
        #     # Add penalization in the end of the episode?
        # else:
        #     done = False

        # Get the next state
        self._next_state()
        
        # Update stepper
        self.episode_step += 1
        self.current_step +=1

        # Return gym env step structure
        return self.state, reward, done , {}

    def render(self):
        #img = self.visualization.render(self.df.iloc[self.current_step], self.net_worth, self.trades)
        #return img
        return False

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

        # Define how much will invest
        self._set_initial_investment(self.dataset_idx)
        self._set_episode_target(self.dataset_idx)
    
        # Environment complete
        self.environment_beaten = False

        # # State queues
        self.orders_history = deque(maxlen=self.lookback)    # Order tracking during episode
        self.portfolio_history = deque(maxlen=self.lookback) # Portfolio evolution tracking during episode
        self.market_history = deque(maxlen=self.lookback)    # Dataframe data during episode

        # History queues
        self.trades = deque(maxlen=self.window_size)

        # Trader state
        self.portfolio_value = self.initial_investment
        self.prev_portfolio_value = self.initial_investment
        self.cash_in_hand = self.initial_investment
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

        self.episode_history = [] # step, data, ação, preço 
        # self.episode_penalty = False

        # # Stepper
        self.episode_step = 0
        self.current_step = self.lookback

        # # Create the initial state
        #for i in reversed(range(self.lookback)):
        for i in range(self.lookback):
            current_step = self.current_step -i

            # Orders history tracks recent trader activity - held bought sold
            self.orders_history.append([0,0,0]) # Held, Sold, Bought

            # Portfolio
            self.portfolio_history.append([1,1,0,1])  # portfolio_value_%  cash_held_% stocks_held_% stock_price_avg_comp_%

            # Market history 
            self.market_history.append(self.df_norm.iloc[current_step])



        # # Create a new state
        # # Should I make a state object that tracks this shit?
        self._state()

        # return self.state
        return self.state 

    def init_visualization(self):
        self.visualization = TradingGraph(render_range=66, show_reward=True, show_indicators=True) # init visualization

        