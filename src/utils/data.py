import os
import nltk
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from stockstats import StockDataFrame as Sdf

from src.constants import ALLOWED_NEWS_TICKERS,DOW_30_2021


class Downloader:
    def __init__(self, start_date="2017-01-01", end_date= "2020-01-01", tickers=DOW_30_2021):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = tickers

    def read_csv(self,file_path):
        df = pd.read_csv(file_path)
        try:
            key = 'Date'
            df[key] =  pd.to_datetime(df[key])
            df.set_index([key],inplace=True)
        except:
            key = 'date'
            df[key] =  pd.to_datetime(df[key])
            df.set_index([key],inplace=True)
        return df

    def fetch_single_ticker(self, ticker, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date

        if end_date is None:
            end_date = self.end_date

        file_path = 'storage/datasets/ohlc__'+ticker.lower()+'__'+str(start_date)+'__'+str(end_date)+'.csv'
        if not os.path.exists(file_path):
            df = yf.download(
                ticker, start=start_date, end=end_date
            )
            df["ticker"] = ticker.lower()
            pd.DataFrame(df).to_csv(file_path)
        else:
            df = self.read_csv(file_path)

        return df

    def fetch_data(self, tickers=None, start_date=None, end_date=None):
        if tickers is None:
            ticker_list = self.ticker_list
        else:
            ticker_list = tickers

        if start_date is None:
            start_date = self.start_date

        if end_date is None:
            end_date = self.end_date


        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = self.fetch_single_ticker(tic,start_date,end_date)
            temp_df['ticker'] = tic.lower()
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "ticker",
            ]
            # use adjusted close price instead of close price
            #data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            # data_df = data_df.drop(labels="adjcp", axis=1)
            data_df = data_df.drop(labels="adj_close", axis=1)
            
        except NotImplementedError:
            print("the features are not supported currently")

        # create day of the week column (monday = 0)
        data_df['date'] = pd.to_datetime(data_df['date'])
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["ticker", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.ticker.value_counts() >= mean_df)
        names = df.ticker.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.ticker.isin(select_stocks_list)]
        return df

class FeatureEngeneer:
    def __init__(self,df):
        self.df = df

    def _parse_df_dates(self,df):
        start_date = df.date.min()
        end_date = df.date.max()
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_date = datetime.strftime(end_date, "%Y-%m-%d")

        return start_date, end_date

    def _get_indicator_file_name(self, ticker, indicator, start_date, end_date):
        return 'storage/datasets/indicators__'+ticker.lower()+'__'+indicator+'__'+str(start_date)+'__'+str(end_date)+'.csv'

    def cleanup(self,df):
        return df.dropna().reset_index(drop=True)

    def fear_and_greed(self,df):
        df = df.copy()
        df['date'] = pd.to_datetime(df.date).dt.date
        fg_df = pd.read_csv('storage/datasets/fear_greed.csv')
        #fg_df.rename(columns={"Date":"date"},inplace=True)
        fg_df['date'] = pd.to_datetime(fg_df.date).dt.date
        if 'Unnamed: 0' in fg_df.columns:
            fg_df.drop(columns=['Unnamed: 0'],inplace=True)

        return df.merge(fg_df,on="date")

    def trends(self,df,ticker, keys="ticker"):
        ticker = ticker.lower()
        df = df.copy()
        df = df[df['ticker']== ticker]
        df['date'] = pd.to_datetime(df.date).dt.date
        trends_df = pd.read_csv('storage/datasets/trends__'+ticker.lower()+'.csv')
        if keys == "ticker":
            trends_df = trends_df[[ticker,'date']]

        if 'Unnamed: 0' in trends_df.columns:
            trends_df.drop(columns=['Unnamed: 0'],inplace=True)

        trends_df['date'] = pd.to_datetime(trends_df.date).dt.date

        return df.merge(trends_df,on="date")

    def sentiment_analysis(self,df,ticker, key="title", tool="vader"):
        #assert ticker.upper() in ALLOWED_NEWS_TICKERS
        ticker = ticker.lower()

        df = df.copy()
        df = df[df['ticker']== ticker]
        df['date'] = pd.to_datetime(df.date).dt.date

        if key == "title":
            news_df = pd.read_csv('storage/datasets/news_header_sentiment_analysis__'+ticker.lower()+'__'+tool+'.csv')
        else:
            news_df = pd.read_csv('storage/datasets/news_body_sentiment_analysis__'+ticker.lower()+'__'+tool+'.csv')

        if 'Unnamed: 0' in news_df.columns:
            news_df.drop(columns=['Unnamed: 0'],inplace=True)

        grouped_news = news_df.groupby(by="date").mean()
        grouped_news.reset_index(inplace=True)
        grouped_news['date'] = pd.to_datetime(grouped_news.date).dt.date

        grouped_news.rename(columns={"neg":"neg_"+key,"pos":"pos_"+key,"neu":"neu_"+key,"compound":"compound_"+key},inplace=True)

        return df.merge(grouped_news,on="date")

        #return df


    def technical_indicators(self,df=None, technical_indicators=[
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]):
        if df is None:
            df = self.df


        df = df.copy()
        df = df.sort_values(by=["ticker", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.ticker.unique()

        start_date,end_date = self._parse_df_dates(df)

        for indicator in technical_indicators:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    indicator_file_path = self._get_indicator_file_name(unique_ticker[i],indicator,start_date,end_date)
                    if not os.path.exists(indicator_file_path):
                        temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["ticker"] = unique_ticker[i]
                        temp_indicator["date"] = df[df.ticker == unique_ticker[i]][
                            "date"
                        ].to_list()

                        temp_indicator.to_csv(indicator_file_path)

                    else:
                        temp_indicator = Downloader().read_csv(indicator_file_path)
                        #temp_indicator.reset_index(inplace=True)

                        temp_indicator.rename(columns={'date.1':'date'},inplace=True)

                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )

                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["ticker", "date", indicator]], on=["ticker", "date"], how="left"
            )
        df = df.sort_values(by=["date", "ticker"])
        return df

    def dji(self, df):
        if df is None:
            df = self.df

        start_date, end_date = self._parse_df_dates(df)

        df = df.copy()
        df_dji = Downloader().fetch_data(start_date=start_date, end_date=end_date, tickers=["^DJI"])

        return df_dji


    def vix(self,df=None):
        if df is None:
            df = self.df

        start_date, end_date = self._parse_df_dates(df)

        df = df.copy()
        df_vix = Downloader().fetch_data(start_date=start_date, end_date=end_date, tickers=["^VIX"])
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]
        vix['vix'] = vix['vix']/100

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return df

    def turbulence(self, df=None, asset="stock"):
        if df is None:
            df = self.df

        df = df.copy()
        turbulence_index = self.calculate_turbulence(df=df, asset=asset)
        turbulence_index['turbulence'] = turbulence_index['turbulence'] / 500
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, df=None, asset="stock"):

        # Global finance turbulence index should be done with this:
        # U.S. Stocks
        # Non-U.S. Stocks
        # U.S. Bonds
        # Non-U.S. Bonds
        # U.S. Real Estate
        # Non-U.S. Real Estate
        # Commodities
        if df is None:
            df = self.df

        # Should be based on dow 30 or something
        # can add other market assets

        if asset == "stock":
            start = 252
        else:
            start = 356

        df = df.copy()
        df_price_pivot = df.pivot(index="date", columns="ticker", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year

        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - start])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )


            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
            turbulence_index[turbulence_index['turbulence'] == 0] = np.nan
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

class SentimentAnalysis:
    def __init__(self):
        nltk.downloader.download('vader_lexicon')
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        self.vader = SentimentIntensityAnalyzer()

    def evaluate(self, news_df, key="title", tool="vader"):

        columns = ['ticker', 'date', key, 'provider']
        parsed_and_scored_news = pd.DataFrame(news_df, columns=columns)

        if tool=="vader":
            scores = parsed_and_scored_news[key].apply(self.vader.polarity_scores).tolist()

        scores_df = pd.DataFrame(scores)
        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
        parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
        parsed_and_scored_news.sort_values(by="date",inplace=True)
        parsed_and_scored_news.reset_index(drop=True, inplace=True)

        return parsed_and_scored_news

    def evaluate_title(self,news_df,tool="vader"):
        return self.evaluate(news_df,key="title",tool=tool)

    def evaluate_content(self,news_df,tool="vader"):
        return self.evaluate(news_df,key="content",tool=tool)


class FundamentalData:
    def __init__(self):
        pass


    # Import fundamental data from my GitHub repository
    # ['gvkey',
 # 'datadate',
 # 'fyearq',
 # 'fqtr',
 # 'fyr',
 # 'indfmt',
 # 'consol',
 # 'popsrc',
 # 'datafmt',
 # 'tic',
 # 'conm',
 # 'acctchgq',
 # 'acctstdq',
 # 'adrrq',
 # 'ajexq',
 # 'ajpq',
 # 'bsprq',
 # 'compstq',
 # 'curcdq',
 # 'curncdq',
 # 'currtrq',
 # 'curuscnq',
 # 'datacqtr',
 # 'datafqtr',
 # 'finalq',
 # 'ogmq',
 # 'rp',
 # 'scfq',
 # 'srcq',
 # 'staltq',
 # 'updq',
 # 'apdedateq',
 # 'fdateq',
 # 'pdateq',
 # 'rdq',
 # 'acchgq',
 # 'acomincq',
 # 'acoq',
 # 'actq',
 # 'altoq',
 # 'ancq',
 # 'anoq',
 # 'aociderglq',
 # 'aociotherq',
 # 'aocipenq',
 # 'aocisecglq',
 # 'aol2q',
 # 'aoq',
 # 'apq',
 # 'aqaq',
 # 'aqdq',
 # 'aqepsq',
 # 'aqpl1q',
 # 'aqpq',
 # 'arcedq',
 # 'arceepsq',
 # 'arceq',
 # 'atq',
 # 'aul3q',
 # 'billexceq',
 # 'capr1q',
 # 'capr2q',
 # 'capr3q',
 # 'capsftq',
 # 'capsq',
 # 'ceiexbillq',
 # 'ceqq',
 # 'cheq',
 # 'chq',
 # 'cibegniq',
 # 'cicurrq',
 # 'ciderglq',
 # 'cimiiq',
 # 'ciotherq',
 # 'cipenq',
 # 'ciq',
 # 'cisecglq',
 # 'citotalq',
 # 'cogsq',
 # 'csh12q',
 # 'cshfd12',
 # 'cshfdq',
 # 'cshiq',
 # 'cshopq',
 # 'cshoq',
 # 'cshprq',
 # 'cstkcvq',
 # 'cstkeq',
 # 'cstkq',
 # 'dcomq',
 # 'dd1q',
 # 'deracq',
 # 'deraltq',
 # 'derhedglq',
 # 'derlcq',
 # 'derlltq',
 # 'diladq',
 # 'dilavq',
 # 'dlcq',
 # 'dlttq',
 # 'doq',
 # 'dpacreq',
 # 'dpactq',
 # 'dpq',
 # 'dpretq',
 # 'drcq',
 # 'drltq',
 # 'dteaq',
 # 'dtedq',
 # 'dteepsq',
 # 'dtepq',
 # 'dvintfq',
 # 'dvpq',
 # 'epsf12',
 # 'epsfi12',
 # 'epsfiq',
 # 'epsfxq',
 # 'epspi12',
 # 'epspiq',
 # 'epspxq',
 # 'epsx12',
 # 'esopctq',
 # 'esopnrq',
 # 'esoprq',
 # 'esoptq',
 # 'esubq',
 # 'fcaq',
 # 'ffoq',
 # 'finacoq',
 # 'finaoq',
 # 'finchq',
 # 'findlcq',
 # 'findltq',
 # 'finivstq',
 # 'finlcoq',
 # 'finltoq',
 # 'finnpq',
 # 'finreccq',
 # 'finrecltq',
 # 'finrevq',
 # 'finxintq',
 # 'finxoprq',
 # 'gdwlamq',
 # 'gdwlia12',
 # 'gdwliaq',
 # 'gdwlid12',
 # 'gdwlidq',
 # 'gdwlieps12',
 # 'gdwliepsq',
 # 'gdwlipq',
 # 'gdwlq',
 # 'glaq',
 # 'glcea12',
 # 'glceaq',
 # 'glced12',
 # 'glcedq',
 # 'glceeps12',
 # 'glceepsq',
 # 'glcepq',
 # 'gldq',
 # 'glepsq',
 # 'glivq',
 # 'glpq',
 # 'hedgeglq',
 # 'ibadj12',
 # 'ibadjq',
 # 'ibcomq',
 # 'ibmiiq',
 # 'ibq',
 # 'icaptq',
 # 'intaccq',
 # 'intanoq',
 # 'intanq',
 # 'invfgq',
 # 'invoq',
 # 'invrmq',
 # 'invtq',
 # 'invwipq',
 # 'ivaeqq',
 # 'ivaoq',
 # 'ivltq',
 # 'ivstq',
 # 'lcoq',
 # 'lctq',
 # 'lltq',
 # 'lnoq',
 # 'lol2q',
 # 'loq',
 # 'loxdrq',
 # 'lqpl1q',
 # 'lseq',
 # 'ltmibq',
 # 'ltq',
 # 'lul3q',
 # 'mibnq',
 # 'mibq',
 # 'mibtq',
 # 'miiq',
 # 'msaq',
 # 'ncoq',
 # 'niitq',
 # 'nimq',
 # 'niq',
 # 'nopiq',
 # 'npatq',
 # 'npq',
 # 'nrtxtdq',
 # 'nrtxtepsq',
 # 'nrtxtq',
 # 'obkq',
 # 'oepf12',
 # 'oeps12',
 # 'oepsxq',
 # 'oiadpq',
 # 'oibdpq',
 # 'opepsq',
 # 'optdrq',
 # 'optfvgrq',
 # 'optlifeq',
 # 'optrfrq',
 # 'optvolq',
 # 'piq',
 # 'pllq',
 # 'pnc12',
 # 'pncd12',
 # 'pncdq',
 # 'pnceps12',
 # 'pncepsq',
 # 'pnciapq',
 # 'pnciaq',
 # 'pncidpq',
 # 'pncidq',
 # 'pnciepspq',
 # 'pnciepsq',
 # 'pncippq',
 # 'pncipq',
 # 'pncpd12',
 # 'pncpdq',
 # 'pncpeps12',
 # 'pncpepsq',
 # 'pncpq',
 # 'pncq',
 # 'pncwiapq',
 # 'pncwiaq',
 # 'pncwidpq',
 # 'pncwidq',
 # 'pncwiepq',
 # 'pncwiepsq',
 # 'pncwippq',
 # 'pncwipq',
 # 'pnrshoq',
 # 'ppegtq',
 # 'ppentq',
 # 'prcaq',
 # 'prcd12',
 # 'prcdq',
 # 'prce12',
 # 'prceps12',
 # 'prcepsq',
 # 'prcpd12',
 # 'prcpdq',
 # 'prcpeps12',
 # 'prcpepsq',
 # 'prcpq',
 # 'prcraq',
 # 'prshoq',
 # 'pstknq',
 # 'pstkq',
 # 'pstkrq',
 # 'rcaq',
 # 'rcdq',
 # 'rcepsq',
 # 'rcpq',
 # 'rdipaq',
 # 'rdipdq',
 # 'rdipepsq',
 # 'rdipq',
 # 'recdq',
 # 'rectaq',
 # 'rectoq',
 # 'rectq',
 # 'rectrq',
 # 'recubq',
 # 'req',
 # 'retq',
 # 'reunaq',
 # 'revtq',
 # 'rllq',
 # 'rra12',
 # 'rraq',
 # 'rrd12',
 # 'rrdq',
 # 'rreps12',
 # 'rrepsq',
 # 'rrpq',
 # 'rstcheltq',
 # 'rstcheq',
 # 'saleq',
 # 'seqoq',
 # 'seqq',
 # 'seta12',
 # 'setaq',
 # 'setd12',
 # 'setdq',
 # 'seteps12',
 # 'setepsq',
 # 'setpq',
 # 'spce12',
 # 'spced12',
 # 'spcedpq',
 # 'spcedq',
 # 'spceeps12',
 # 'spceepsp12',
 # 'spceepspq',
 # 'spceepsq',
 # 'spcep12',
 # 'spcepd12',
 # 'spcepq',
 # 'spceq',
 # 'spidq',
 # 'spiepsq',
 # 'spioaq',
 # 'spiopq',
 # 'spiq',
 # 'sretq',
 # 'stkcoq',
 # 'stkcpaq',
 # 'teqq',
 # 'tfvaq',
 # 'tfvceq',
 # 'tfvlq',
 # 'tieq',
 # 'tiiq',
 # 'tstknq',
 # 'tstkq',
 # 'txdbaq',
 # 'txdbcaq',
 # 'txdbclq',
 # 'txdbq',
 # 'txdiq',
 # 'txditcq',
 # 'txpq',
 # 'txtq',
 # 'txwq',
 # 'uacoq',
 # 'uaoq',
 # 'uaptq',
 # 'ucapsq',
 # 'ucconsq',
 # 'uceqq',
 # 'uddq',
 # 'udmbq',
 # 'udoltq',
 # 'udpcoq',
 # 'udvpq',
 # 'ugiq',
 # 'uinvq',
 # 'ulcoq',
 # 'uniamiq',
 # 'unopincq',
 # 'uopiq',
 # 'updvpq',
 # 'upmcstkq',
 # 'upmpfq',
 # 'upmpfsq',
 # 'upmsubpq',
 # 'upstkcq',
 # 'upstkq',
 # 'urectq',
 # 'uspiq',
 # 'usubdvpq',
 # 'usubpcvq',
 # 'utemq',
 # 'wcapq',
 # 'wdaq',
 # 'wddq',
 # 'wdepsq',
 # 'wdpq',
 # 'xaccq',
 # 'xidoq',
 # 'xintq',
 # 'xiq',
 # 'xoprq',
 # 'xopt12',
 # 'xoptd12',
 # 'xoptd12p',
 # 'xoptdq',
 # 'xoptdqp',
 # 'xopteps12',
 # 'xoptepsp12',
 # 'xoptepsq',
 # 'xoptepsqp',
 # 'xoptq',
 # 'xoptqp',
 # 'xrdq',
 # 'xsgaq',
 # 'acchgy',
 # 'afudccy',
 # 'afudciy',
 # 'amcy',
 # 'aolochy',
 # 'apalchy',
 # 'aqay',
 # 'aqcy',
 # 'aqdy',
 # 'aqepsy',
 # 'aqpy',
 # 'arcedy',
 # 'arceepsy',
 # 'arcey',
 # 'capxy',
 # 'cdvcy',
 # 'chechy',
 # 'cibegniy',
 # 'cicurry',
 # 'cidergly',
 # 'cimiiy',
 # 'ciothery',
 # 'cipeny',
 # 'cisecgly',
 # 'citotaly',
 # 'ciy',
 # 'cogsy',
 # 'cshfdy',
 # 'cshpry',
 # 'cstkey',
 # 'depcy',
 # 'derhedgly',
 # 'dilady',
 # 'dilavy',
 # 'dlcchy',
 # 'dltisy',
 # 'dltry',
 # 'doy',
 # 'dpcy',
 # 'dprety',
 # 'dpy',
 # 'dteay',
 # 'dtedy',
 # 'dteepsy',
 # 'dtepy',
 # 'dvpy',
 # 'dvy',
 # 'epsfiy',
 # 'epsfxy',
 # 'epspiy',
 # 'epspxy',
 # 'esubcy',
 # 'esuby',
 # 'exrey',
 # 'fcay',
 # 'ffoy',
 # 'fiaoy',
 # 'fincfy',
 # 'finrevy',
 # 'finxinty',
 # 'finxopry',
 # 'fopoxy',
 # 'fopoy',
 # 'fopty',
 # 'fsrcoy',
 # 'fsrcty',
 # 'fuseoy',
 # 'fusety',
 # 'gdwlamy',
 # 'gdwliay',
 # 'gdwlidy',
 # 'gdwliepsy',
 # 'gdwlipy',
 # 'glay',
 # 'glceay',
 # 'glcedy',
 # 'glceepsy',
 # 'glcepy',
 # 'gldy',
 # 'glepsy',
 # 'glivy',
 # 'glpy',
 # 'hedgegly',
 # 'ibadjy',
 # 'ibcomy',
 # 'ibcy',
 # 'ibmiiy',
 # 'iby',
 # 'intpny',
 # 'invchy',
 # 'itccy',
 # 'ivacoy',
 # 'ivchy',
 # 'ivncfy',
 # 'ivstchy',
 # 'miiy',
 # 'ncoy',
 # 'niity',
 # 'nimy',
 # 'niy',
 # 'nopiy',
 # 'nrtxtdy',
 # 'nrtxtepsy',
 # 'nrtxty',
 # 'oancfy',
 # 'oepsxy',
 # 'oiadpy',
 # 'oibdpy',
 # 'opepsy',
 # 'optdry',
 # 'optfvgry',
 # 'optlifey',
 # 'optrfry',
 # 'optvoly',
 # 'pdvcy',
 # 'piy',
 # 'plly',
 # 'pncdy',
 # 'pncepsy',
 # 'pnciapy',
 # 'pnciay',
 # 'pncidpy',
 # 'pncidy',
 # 'pnciepspy',
 # 'pnciepsy',
 # 'pncippy',
 # 'pncipy',
 # 'pncpdy',
 # 'pncpepsy',
 # 'pncpy',
 # 'pncwiapy',
 # 'pncwiay',
 # 'pncwidpy',
 # 'pncwidy',
 # 'pncwiepsy',
 # 'pncwiepy',
 # 'pncwippy',
 # 'pncwipy',
 # 'pncy',
 # 'prcay',
 # 'prcdy',
 # 'prcepsy',
 # 'prcpdy',
 # 'prcpepsy',
 # 'prcpy',
 # 'prstkccy',
 # 'prstkcy',
 # 'prstkpcy',
 # 'rcay',
 # 'rcdy',
 # 'rcepsy',
 # 'rcpy',
 # 'rdipay',
 # 'rdipdy',
 # 'rdipepsy',
 # 'rdipy',
 # 'recchy',
 # 'revty',
 # 'rray',
 # 'rrdy',
 # 'rrepsy',
 # 'rrpy',
 # 'saley',
 # 'scstkcy',
 # 'setay',
 # 'setdy',
 # 'setepsy',
 # 'setpy',
 # 'sivy',
 # 'spcedpy',
 # 'spcedy',
 # 'spceepspy',
 # 'spceepsy',
 # 'spcepy',
 # 'spcey',
 # 'spidy',
 # 'spiepsy',
 # 'spioay',
 # 'spiopy',
 # 'spiy',
 # 'sppey',
 # 'sppivy',
 # 'spstkcy',
 # 'srety',
 # 'sstky',
 # 'stkcoy',
 # 'stkcpay',
 # 'tdcy',
 # 'tfvcey',
 # 'tiey',
 # 'tiiy',
 # 'tsafcy',
 # 'txachy',
 # 'txbcofy',
 # 'txbcoy',
 # 'txdcy',
 # 'txdiy',
 # 'txpdy',
 # 'txty',
 # 'txwy',
 # 'uaolochy',
 # 'udfccy',
 # 'udvpy',
 # 'ufretsdy',
 # 'ugiy',
 # 'uniamiy',
 # 'unopincy',
 # 'unwccy',
 # 'uoisy',
 # 'updvpy',
 # 'uptacy',
 # 'uspiy',
 # 'ustdncy',
 # 'usubdvpy',
 # 'utfdocy',
 # 'utfoscy',
 # 'utmey',
 # 'uwkcapcy',
 # 'wcapchy',
 # 'wcapcy',
 # 'wday',
 # 'wddy',
 # 'wdepsy',
 # 'wdpy',
 # 'xidocy',
 # 'xidoy',
 # 'xinty',
 # 'xiy',
 # 'xopry',
 # 'xoptdqpy',
 # 'xoptdy',
 # 'xoptepsqpy',
 # 'xoptepsy',
 # 'xoptqpy',
 # 'xopty',
 # 'xrdy',
 # 'xsgay',
 # 'exchg',
 # 'costat',
 # 'cshtrq',
 # 'dvpspq',
 # 'dvpsxq',
 # 'mkvaltq',
 # 'prccq',
 # 'prchq',
 # 'prclq',
 # 'adjex',
 # 'ggroup',
 # 'gind',
 # 'gsector',
 # 'gsubind']


# items = [
#     'datadate', # Date
#     'tic', # Ticker
#     'oiadpq', # Quarterly operating income
#     'revtq', # Quartely revenue
#     'niq', # Quartely net income
#     'atq', # Total asset
#     'teqq', # Shareholder's equity
#     'epspiy', # EPS(Basic) incl. Extraordinary items
#     'ceqq', # Common Equity
#     'cshoq', # Common Shares Outstanding
#     'dvpspq', # Dividends per share
#     'actq', # Current assets
#     'lctq', # Current liabilities
#     'cheq', # Cash & Equivalent
#     'rectq', # Recievalbles
#     'cogsq', # Cost of  Goods Sold
#     'invtq', # Inventories
#     'apq',# Account payable
#     'dlttq', # Long term debt
#     'dlcq', # Debt in current liabilites
#     'ltq' # Liabilities
# ]

# # Rename column names for the sake of readability
# fund_data = fund_data.rename(columns={
#     'datadate':'date', # Date
#     'oiadpq':'op_inc_q', # Quarterly operating income
#     'revtq':'rev_q', # Quartely revenue
#     'niq':'net_inc_q', # Quartely net income
#     'atq':'tot_assets', # Assets
#     'teqq':'sh_equity', # Shareholder's equity
#     'epspiy':'eps_incl_ex', # EPS(Basic) incl. Extraordinary items
#     'ceqq':'com_eq', # Common Equity
#     'cshoq':'sh_outstanding', # Common Shares Outstanding
#     'dvpspq':'div_per_sh', # Dividends per share
#     'actq':'cur_assets', # Current assets
#     'lctq':'cur_liabilities', # Current liabilities
#     'cheq':'cash_eq', # Cash & Equivalent
#     'rectq':'receivables', # Receivalbles
#     'cogsq':'cogs_q', # Cost of  Goods Sold
#     'invtq':'inventories', # Inventories
#     'apq': 'payables',# Account payable
#     'dlttq':'long_debt', # Long term debt
#     'dlcq':'short_debt', # Debt in current liabilites
#     'ltq':'tot_liabilities' # Liabilities
# })

# 4.3 Calculate financial ratios
# For items from Profit/Loss statements, we calculate LTM (Last Twelve Months) and use them to derive profitability related ratios such as Operating Maring and ROE. For items from balance sheets, we use the numbers on the day.
# To check the definitions of the financial ratios calculated here, please refer to CFI's website: https://corporatefinanceinstitute.com/resources/knowledge/finance/financial-ratios/
# # Calculate financial ratios
# date = pd.to_datetime(fund_data['date'],format='%Y%m%d')
#
# tic = fund_data['tic'].to_frame('tic')
#
# # Profitability ratios
# # Operating Margin
# OPM = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='OPM')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         OPM[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         OPM.iloc[i] = np.nan
#     else:
#         OPM.iloc[i] = np.sum(fund_data['op_inc_q'].iloc[i-3:i])/np.sum(fund_data['rev_q'].iloc[i-3:i])
#
# # Net Profit Margin
# NPM = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='NPM')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         NPM[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         NPM.iloc[i] = np.nan
#     else:
#         NPM.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/np.sum(fund_data['rev_q'].iloc[i-3:i])
#
# # Return On Assets
# ROA = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='ROA')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         ROA[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         ROA.iloc[i] = np.nan
#     else:
#         ROA.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/fund_data['tot_assets'].iloc[i]
#
# # Return on Equity
# ROE = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='ROE')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         ROE[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         ROE.iloc[i] = np.nan
#     else:
#         ROE.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i-3:i])/fund_data['sh_equity'].iloc[i]
#
# # For calculating valuation ratios in the next subpart, calculate per share items in advance
# # Earnings Per Share
# EPS = fund_data['eps_incl_ex'].to_frame('EPS')
#
# # Book Per Share
# BPS = (fund_data['com_eq']/fund_data['sh_outstanding']).to_frame('BPS') # Need to check units
#
# #Dividend Per Share
# DPS = fund_data['div_per_sh'].to_frame('DPS')
#
# # Liquidity ratios
# # Current ratio
# cur_ratio = (fund_data['cur_assets']/fund_data['cur_liabilities']).to_frame('cur_ratio')
#
# # Quick ratio
# quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables'] )/fund_data['cur_liabilities']).to_frame('quick_ratio')
#
# # Cash ratio
# cash_ratio = (fund_data['cash_eq']/fund_data['cur_liabilities']).to_frame('cash_ratio')


# # Efficiency ratios
# # Inventory turnover ratio
# inv_turnover = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='inv_turnover')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         inv_turnover[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         inv_turnover.iloc[i] = np.nan
#     else:
#         inv_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i-3:i])/fund_data['inventories'].iloc[i]
#
# # Receivables turnover ratio
# acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='acc_rec_turnover')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         acc_rec_turnover[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         acc_rec_turnover.iloc[i] = np.nan
#     else:
#         acc_rec_turnover.iloc[i] = np.sum(fund_data['rev_q'].iloc[i-3:i])/fund_data['receivables'].iloc[i]
#
# # Payable turnover ratio
# acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0],dtype=object),name='acc_pay_turnover')
# for i in range(0, fund_data.shape[0]):
#     if i-3 < 0:
#         acc_pay_turnover[i] = np.nan
#     elif fund_data.iloc[i,1] != fund_data.iloc[i-3,1]:
#         acc_pay_turnover.iloc[i] = np.nan
#     else:
#         acc_pay_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i-3:i])/fund_data['payables'].iloc[i]
#
# ## Leverage financial ratios
# # Debt ratio
# debt_ratio = (fund_data['tot_liabilities']/fund_data['tot_assets']).to_frame('debt_ratio')
#
# # Debt to Equity ratio
# debt_to_equity = (fund_data['tot_liabilities']/fund_data['sh_equity']).to_frame('debt_to_equity')


# # Create a dataframe that merges all the ratios
# ratios = pd.concat([date,tic,OPM,NPM,ROA,ROE,EPS,BPS,DPS,
#                     cur_ratio,quick_ratio,cash_ratio,inv_turnover,acc_rec_turnover,acc_pay_turnover,
#                    debt_ratio,debt_to_equity], axis=1)
