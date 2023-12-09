from download import *
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import datetime as dt
from sklearn.cluster import KMeans
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
    data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
    return data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

def calculate_returns(data_df):
    outlier_cutoff = 0.005
    lags = [1,2,3,6,9,12]
    for lag in lags:
        data_df[f'return_{lag}m'] = (data_df['adj close']
                                     .pct_change(lag)
                                     .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff)))
                                     .add(1).pow(1/lag).sub(1))
    return data_df

def get_fama_french_factors():
    # get monthly factors
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                   'famafrench',start='2010')[0].drop('RF', axis=1)
    factor_data.index = factor_data.index.to_timestamp()
    
    # Change date to be at the last of every month and convert factors to decimals from percentages
    factor_data = factor_data.resample('M').last().div(100)
    factor_data.index.name = 'date'
    return factor_data

def calculate_rolling_basis(factor_data):
    factor_data = (factor_data.groupby(level=1,
                                       group_keys=False)
     .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                 exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                 window=min(24, x.shape[0]),
                                 min_nobs=len(x.columns) + 1)
     .fit(params_only=True)
     .params
     .drop('const', axis=1)))
    factor_data = factor_data.groupby('ticker').shift()
    return factor_data
    
def make_clusters(data):
    initial_centroids = initialize_centroids()
    data['cluster'] = KMeans(n_clusters=4, random_state=0,init=initial_centroids).fit(data).labels_
    return data

def plot_clusters(data):
    cluster_0 = data[data['cluster'] == 0]
    cluster_1 = data[data['cluster'] == 1]
    cluster_2 = data[data['cluster'] == 2]
    cluster_3 = data[data['cluster'] == 3]
    
    plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6], color='red', label='Cluster 0')
    plt.scatter(cluster_1.iloc[:,0], cluster_1.iloc[:,6], color='green', label='Cluster 1')
    plt.scatter(cluster_2.iloc[:,0], cluster_2.iloc[:,6], color='blue', label='Cluster 2')
    plt.scatter(cluster_3.iloc[:,0], cluster_3.iloc[:,6], color='black', label='Cluster 3')
    
    plt.legend()
    plt.show()
    return


# Make clusters consistent across all rsi values (i.e. cluster 3 always assigned to 55 <= rsi)
def initialize_centroids():
    target_rsi_values = [30, 45, 55, 70]
    initial_centroids = np.zeros((len(target_rsi_values), 18))
    initial_centroids[:, 6] = target_rsi_values
    return initial_centroids

def main():
    sp500_df = get_SP500()
    indicator_calculator = IndicatorCalculator(sp500_df=sp500_df)
    sp500_df = indicator_calculator.calculate_simple_indicators(sp500_df=sp500_df)
    # sp500_df['atr'] = sp500_df.groupby(level=1, group_keys=False).apply(calculate_atr)
    # sp500_df['macd'] = sp500_df.groupby(level=1, group_keys=False)['adj close'].apply(calculate_macd)
    aggregate_data = aggregate(sp500_df=sp500_df)
    aggregate_data = filter_150(aggregate_data)
    aggregate_data = aggregate_data.groupby(level=1,group_keys=False).apply(calculate_returns).dropna()
    # print(aggregate_data.columns.to_list())
    # print(aggregate_data)
    factor_data = get_fama_french_factors()
    factor_data = factor_data.join(aggregate_data['return_1m']).sort_index()
    
    
    # Filter out stocks with less than 10 months of data
    observations = factor_data.groupby(level=1).size()
    valid_stocks = observations[observations > 10]
    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]
    factor_data = calculate_rolling_basis(factor_data=factor_data)
    aggregate_data = aggregate_data.join(factor_data.groupby('ticker').shift())
    
    # Fillin all NaN values with mean of ticker
    factors = ['Mkt-RF', "SMB", "HML", "RMW", "CMA"]
    aggregate_data.loc[:, factors] = aggregate_data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
    aggregate_data = aggregate_data.drop('adj close', axis=1)
    aggregate_data = aggregate_data.dropna()
    
    aggregate_data = aggregate_data.dropna().groupby('date', group_keys=False).apply(make_clusters)
    
    plt.style.use('ggplot')
    for i in aggregate_data.index.get_level_values('date').unique().tolist():
        level = aggregate_data.xs(i, level=0)
        plt.title(f'Date {i}')
        plot_clusters(level)
    
if __name__ == "__main__":
    main()