import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

def get_SP500():
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S&P_500_companies')[0]
    sp500['Symbol'] = sp500['Symbol'].str.replace('.','-') # Cleaning up symbols
    symbols = sp500['Symbol'].unique().tolist()
    end_date = '2023-12-06'
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)
    sp500_df = yf.download(tickers=symbols, start=start_date,end=end_date)
    sp500_df = sp500_df.stack()
    sp500_df.index.names = ['date', 'ticker']
    sp500_df.columns = sp500_df.columns.str.lower()
    return sp500_df   
