import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates

from datetime import datetime
from collections import deque
from mplfinance.original_flavor import candlestick_ohlc

# @TODO: Add indicators list
class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # date, open, high, low, close, volume, portfolio_value, trading_history
    # call render every step
    def __init__(self, render_range=44, show_reward=True, show_indicators=False):
        self.volume = deque(maxlen=render_range)
        self.portfolio_value = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range)
        self.price_data = deque(maxlen=render_range)
        self.render_range = render_range
        self.show_reward = show_reward
        self.show_indicators = show_indicators

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        #plt.style.use('dark_background')

        # close all plots if there are open
        plt.close('all')

        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

        # Add paddings to make graph easier to view
        plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        # @TODO define if show indicators
        # if self.show_indicators:
        #
        #     self.create_indicators_lists()

    # def create_indicators_lists(self):
    #     # Create a new axis for indicatorswhich shares its x-axis with volume
    #     self.ax4 = self.ax2.twinx()
    #
    #     self.sma7 = deque(maxlen=self.render_range)
    #     self.RSI = deque(maxlen=self.render_range)

    # def plot_indicators(self, df, date_render_range):
    #     self.sma7.append(df["sma7"])
 
    #     # Add Simple Moving Average
    #     self.ax1.plot(date_render_range, self.sma7,'-')
 
    #     # # Add Relative Strength Index
    #     self.ax4.plot(date_render_range, self.RSI,'g-')

    def plot_price(self,df,date_render_range):
        self.price_data.append(df["close"])
        self.ax1.plot(date_render_range, self.price_data,'-',color="black",alpha=0.4)


    # Render the environment to the screen
    def render(self, df, portfolio_value, trading_history):
        date = df.name #df["date"]
        open = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # append volume and portfolio_value to deque list
        self.volume.append(volume)
        self.portfolio_value.append(portfolio_value)

        # before appending to deque list, need to convert date to special format
        date = mpl_dates.date2num([pd.to_datetime(date)])[0]
        self.render_data.append([date, open, high, low, close])

        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    
        # Put all dates to one list and fill ax2 sublot with volume
        date_render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(date_render_range, self.volume, 0)

        self.plot_price(df, date_render_range)

        # if self.show_indicators:
        #     self.plot_indicators(df, date_render_range)

        # draw our portfolio_value graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(date_render_range, self.portfolio_value, color="blue")

        # beautify the x-labels (Our date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:,1:])
        maximum = np.max(np.array(self.render_data)[:,1:])
        RANGE = maximum - minimum


        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trading_history:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['date'])])[0]
            if trade_date in date_render_range:
                if trade['action'] == 'buy':
                    high_low = trade['low'] - RANGE*0.02
                    ycoords = trade['low'] - RANGE*0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['high'] + RANGE*0.02
                    ycoords = trade['high'] + RANGE*0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")

                if self.show_reward:
                    try:
                        self.ax1.annotate('{0:.2f}'.format(trade['reward']), (trade_date-0.02, high_low), xytext=(trade_date-0.02, ycoords),
                                                   bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
                    except:
                        pass

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        #plt.show(block=False)
        # Necessary to view frames before they are unrendered
        #plt.pause(0.001)

        """Display image with openCV - no interruption"""

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with openCV or any operation you like
        cv2.imshow("Stock Trading Bot",image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        else:
            return img


def OhlcGraph(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["date"] = pd.to_datetime(df.date)
    df["date"] = df["date"].apply(mpl_dates.date2num)

    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    # We are using the style ‘ggplot’
    plt.style.use('ggplot')
    
    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8)) 

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # Add Simple Moving Average
    ax1.plot(df["date"], df_original['sma7'],'-')
    ax1.plot(df["date"], df_original['sma25'],'-')
    ax1.plot(df["date"], df_original['sma99'],'-')

    # Add Bollinger Bands
    ax1.plot(df["date"], df_original['bb_bbm'],'-')
    ax1.plot(df["date"], df_original['bb_bbh'],'-')
    ax1.plot(df["date"], df_original['bb_bbl'],'-')

    # Add Parabolic Stop and Reverse
    ax1.plot(df["date"], df_original['psar'],'.')

    # # Add Moving Average Convergence Divergence
    ax2.plot(df["date"], df_original['MACD'],'-')

    # # Add Relative Strength Index
    ax2.plot(df["date"], df_original['RSI'],'-')

    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))# %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    
    plt.show()