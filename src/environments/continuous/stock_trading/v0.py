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

from src.constants import ACTIONS, FEES
from src.utils.data import Downloader, FeatureEngeneer

#from ...rules import INDICATOR_NORMALIZATION_RULES
#from ...utils.helpers import get_config_md5

ACTION_AREA_SIZE = 0.7
ACTION_DEAD_AREA = [-1+ACTION_AREA_SIZE,1-ACTION_AREA_SIZE] #[-0.3,0.3]

SPEC =  DotWiz()
SPEC.id = 'StockTradingEnvironment-v0'

class StockTradingEnvironment(Env):
    metadata = {'render.modes': ['human']}

    spec = SPEC

    def __init__(self,
                lookback=10, # 2 weeks
                window_size=66, # 3 months
                start_date= False,
                end_date=False,
                train_percentage=0.7,
                use_technical_indicators= [],
                use_sentiment_analysis=True,
                use_cboe_vix = True,
                use_trends = True,
                use_fear_and_greed=True,
                use_market_volatility= "DOW_30",
                ticker="AAPL",
                initial_investment=100000,
                inertness_punishment=0,#0.001,
                fees=FEES,
                seed=314,
        ):

        self.lookback = lookback
        self.window_size = window_size
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
        self.ticker = ticker
        self.initial_investment = initial_investment
        self.intertness_punishment = inertness_punishment
        self.fees = fees
        self._seed = seed

        self.__init_dataset()

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
        raw_name = json.dumps(name_params)
        df_name = hashlib.md5(json.dumps(name_params,sort_keys=True, indent=2).encode('utf-8')).hexdigest()
        df_path = 'src/environments/continuous/stock_trading/datasets/'+df_name

        self.df_name = df_name

        if not os.path.exists(df_path):
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
            # Alguns precisam de ser a dividir por 10

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
            # @TODO

            # Slice dataframes
            n_dataframes = 0
            print(range(self.lookback, len(df)- self.window_size))
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

            pd.DataFrame([[n_dataframes]],columns=['n_dataframes']).to_csv(df_path+'/config.csv')

            # Se nao existir uma pasta com este nome
            # Cria pasta com este nome
            # Normaliza o dataframe
            # Guarda raw e normalizado
            # Parte raw e normalizado em slices de lookback + window_size ( para ter vários cenários possiveis )
            # Criar um config file com a configuração do ambiente e guarda-a nessa pasta
        config = pd.read_csv(df_path+'/config.csv')
        n_dataframes = config.iloc[0].n_dataframes
        self.n_dataframes = n_dataframes
        print(n_dataframes)

        # Verifica quantos ficheiros raw tem a pasta
        # Faz range de 0 a int(train_percentage) - self.train_df_range e that's it, é excusado carregar agora o dataset, carrega no reset
