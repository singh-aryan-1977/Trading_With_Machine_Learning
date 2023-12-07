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

def aggregate(sp500_df):
    original_cols = ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']
    indicator_cols = [c for c in sp500_df.columns.unique(0) if c not in original_cols]
    
    # print(sp500_df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'))
    aggregated_data = pd.concat([sp500_df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'), 
              sp500_df.unstack()[indicator_cols].resample('M').last().stack('ticker')], axis=1).dropna()
    return aggregated_data
    # print(sp500_df)
    # return sp500_df
    
def filter_150(data):
    data['dollar_volume'] = data['dollar_volume'].unstack('ticker').rolling(5*12).mean().stack()
    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
    return data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

def calculate_returns(data_df):
    outlier_cutoff = 0.005
    lags = [1,2,3,6,9,12]
    for lag in lags:
        data_df[f'return_{lag}m'] = (data_df['adj_close']
                                     .pct_change(lag)
                                     .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff)))
                                     .add(1).pow(1/lag).sub(1))
    return data_df

    
def main():
    sp500_df = get_SP500()
    sp500_df = calculate_simple_indicators(sp500_df=sp500_df)
    sp500_df['atr'] = sp500_df.groupby(level=1, group_keys=False).apply(calculate_atr)
    sp500_df['macd'] = sp500_df.groupby(level=1, group_keys=False)['adj close'].apply(calculate_macd)
    aggregate_data = aggregate(sp500_df=sp500_df)
    aggregate_data = filter_150(aggregate_data)
    aggregate_data = aggregate_data.groupby(level=1,group_keys=False).apply(calculate_returns).dropna()
    print(aggregate_data)
    # sp500_df = aggregate(sp500_df=sp500_df)
    # print(sp500_df)
    
    
if __name__ == "__main__":
    main()