from download import *
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import datetime as dt
import yfinance as yf
import pandas_ta
from download import *

class IndicatorCalculator:
    def __init__(self, sp500_df):
        self.sp500_df = sp500_df
        

    # Calculating garman-klass volatility
    def calculate_gk_volatility(self, sp500_df):
        if sp500_df.empty:
            # print("Reached here 2")
            sp500_df = get_SP500()
        gk_vol_first_part = ((np.log(sp500_df['high'])-np.log(sp500_df['low'])) ** 2) // 2
        gk_vol_second_part = (2*np.log(2)-1)*(np.log(sp500_df['adj close']) - np.log(sp500_df['open']))**2
        sp500_df['garman_klass_volatility'] = gk_vol_first_part - gk_vol_second_part
        return sp500_df

    def calculate_rsi(self, sp500_df):
        if sp500_df.empty:
            sp500_df = get_SP500()
        
        # Group by second index (ticker)
        sp500_df['rsi'] = sp500_df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
        
        return sp500_df

        
    def calculate_bollinger_bands(self, sp500_df):
        if sp500_df.empty:
            sp500_df = get_SP500()
            
        # Select the first column foor bb_low
        sp500_df['bb_low'] = sp500_df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
        
        # Select the second column foor bb_low
        sp500_df['bb_mid'] = sp500_df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
        
        # Select the third column foor bb_low
        sp500_df['bb_high'] = sp500_df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])
        
        return sp500_df

    def calculate_dollar_volume(self, sp500_df):
        if sp500_df.empty:
            sp500_df = get_SP500()
        sp500_df['dollar_volume'] = (sp500_df['adj close'] * sp500_df['volume'])/1e6
        return sp500_df
        
    def calculate_simple_indicators(self, sp500_df):
        sp500_df = self.calculate_gk_volatility(sp500_df=sp500_df)
        sp500_df = self.calculate_rsi(sp500_df=sp500_df)
        sp500_df = self.calculate_bollinger_bands(sp500_df=sp500_df)
        sp500_df = self.calculate_dollar_volume(sp500_df=sp500_df)
        sp500_df['atr'] = sp500_df.groupby(level=1, group_keys=False).apply(self.calculate_atr)
        sp500_df['macd'] = sp500_df.groupby(level=1, group_keys=False)['adj close'].apply(self.calculate_macd)
        return sp500_df

    def calculate_atr(self, stock_data):
        atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'],close=stock_data['close'],length=14)
        return atr.sub(atr.mean()).div(atr.std())

    def calculate_macd(self, close):
            macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
            return macd.sub(macd.mean()).div(macd.std())
    