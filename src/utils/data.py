import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from stockstats import StockDataFrame as Sdf

class Downloader:
    def __init__(self, start_date="2018-01-01", end_date= "2022-01-01", tickers=[]):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = tickers

    def fetch_single_ticker(self, ticker, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date

        if end_date is None:
            end_date = self.end_date

        file_path = 'storage/datasets/ohlc__'+ticker+'__'+str(start_date)+'__'+str(end_date)+'.csv'
        if not os.path.exists(file_path):
            df = yf.download(
                ticker, start=start_date, end=end_date
            )
            df["ticker"] = ticker
            pd.DataFrame(df).to_csv(file_path)
        else:
            df = pd.read_csv(file_path)
            df['Date'] =  pd.to_datetime(df['Date'])
            df.set_index(['Date'],inplace=True)

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
            temp_df['ticker'] = tic
            # temp_df = yf.download(
            #     tic, start=start_date, end=end_date
            # )
            # temp_df["ticker"] = tic
            data_df = data_df.append(temp_df)
            print(temp_df.columns)
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
                "adjcp",
                "volume",
                "ticker",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")

        # create day of the week column (monday = 0)
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
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

class FeatureEngeneer:
    def __init__(self,df):
        self.df = df

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

        for indicator in technical_indicators:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.ticker == unique_ticker[i]][
                        "date"
                    ].to_list()
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

    def vix(self,df=None):
        if df is None:
            df = self.df

        start_date = df.date.min()
        end_date = df.date.max()
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date + timedelta(days=1)
        end_date = datetime.strftime(end_date, "%Y-%m-%d")

        print(df.date.max(),end_date)
        df = df.copy()
        df_vix = Downloader().fetch_data(start_date=start_date, end_date=end_date, tickers=["^VIX"])
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        return df

    def turbulence(self, df=None, asset="stock"):
        if df is None:
            df = self.df

        df = df.copy()
        turbulence_index = self.calculate_turbulence(df=df, asset=asset)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, df=None, asset="stock"):
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
                & (df_price_pivot.index >= unique_date[i - 252])
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
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
