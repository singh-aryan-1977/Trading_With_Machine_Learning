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
from indicators import *

def main():
    sp500_df = get_SP500()
    sp500_df = calculate_simple_indicators(sp500_df=sp500_df)
    sp500_df['atr'] = sp500_df.groupby(level=1, group_keys=False).apply(calculate_atr)
    sp500_df['macd'] = sp500_df.groupby(level=1, group_keys=False)['adj close'].apply(calculate_macd)
    print(sp500_df)
    
    
if __name__ == "__main__":
    main()