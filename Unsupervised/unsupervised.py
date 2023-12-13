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
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from indicators import *
plt.style.use('ggplot')


class UnsupervisedModel():
    def __init__(self, sp500_df):
        self.sp500_df = sp500_df
        
    def aggregate(self, sp500_df):
        original_cols = ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']
        indicator_cols = [c for c in sp500_df.columns.unique(0) if c not in original_cols]
        
        # print(sp500_df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'))
        aggregated_data = pd.concat([sp500_df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'), 
                sp500_df.unstack()[indicator_cols].resample('M').last().stack('ticker')], axis=1).dropna()
        return aggregated_data
        # print(sp500_df)
        # return sp500_df
        
    def filter_150(self, data):
        data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
        data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
        return data[data['dollar_vol_rank'] < 150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

    def calculate_returns(self, data_df):
        outlier_cutoff = 0.005
        lags = [1,2,3,6,9,12]
        for lag in lags:
            data_df[f'return_{lag}m'] = (data_df['adj close']
                                        .pct_change(lag)
                                        .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff)))
                                        .add(1).pow(1/lag).sub(1))
        return data_df

    def get_fama_french_factors(self):
        # get monthly factors
        factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                    'famafrench',start='2010')[0].drop('RF', axis=1)
        factor_data.index = factor_data.index.to_timestamp()
        
        # Change date to be at the last of every month and convert factors to decimals from percentages
        factor_data = factor_data.resample('M').last().div(100)
        factor_data.index.name = 'date'
        return factor_data

    def calculate_rolling_basis(self, factor_data):
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
        
    def make_clusters(self, data):
        initial_centroids = self.initialize_centroids()
        data['cluster'] = KMeans(n_clusters=4, random_state=0,init=initial_centroids).fit(data).labels_
        return data

    def plot_clusters(self, data):
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
    def initialize_centroids(self):
        target_rsi_values = [30, 45, 55, 70]
        initial_centroids = np.zeros((len(target_rsi_values), 18))
        initial_centroids[:, 6] = target_rsi_values
        return initial_centroids

    def plot(self, aggregate_data):
        for i in aggregate_data.index.get_level_values('date').unique().tolist():
            level = aggregate_data.xs(i, level=0)
            plt.title(f'Date {i}')
            self.plot_clusters(level)
            
    def optimize_portfolio_weights(self, prices, lower_bound=0):
        returns = expected_returns.mean_historical_return(prices=prices,frequency=252)
        cov = risk_models.sample_cov(prices=prices,frequency=252)
        ef = EfficientFrontier(expected_returns=returns,
                            cov_matrix=cov,
                            weight_bounds=(lower_bound,.2),
                            solver='SCS')
        weights = ef.max_sharpe()
        return ef.clean_weights()
    
    def learn(self):
        aggregate_data = self.aggregate(self.sp500_df)
        aggregate_data = self.filter_150(aggregate_data)
        aggregate_data = aggregate_data.groupby(level=1,group_keys=False).apply(self.calculate_returns).dropna()
        # print(aggregate_data.columns.to_list())
        # print(aggregate_data)
        factor_data = self.get_fama_french_factors()
        factor_data = factor_data.join(aggregate_data['return_1m']).sort_index()
        
        # Filter out stocks with less than 10 months of data
        observations = factor_data.groupby(level=1).size()
        valid_stocks = observations[observations > 10]
        factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]
        factor_data = self.calculate_rolling_basis(factor_data=factor_data)
        aggregate_data = aggregate_data.join(factor_data.groupby('ticker').shift())
        
        # Fillin all NaN values with mean of ticker
        factors = ['Mkt-RF', "SMB", "HML", "RMW", "CMA"]
        aggregate_data.loc[:, factors] = aggregate_data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
        aggregate_data = aggregate_data.drop('adj close', axis=1)
        aggregate_data = aggregate_data.dropna()
        
        aggregate_data = aggregate_data.dropna().groupby('date', group_keys=False).apply(self.make_clusters)
        # plot(aggregate_data)
        
        filtered_df = aggregate_data[aggregate_data['cluster']==3].copy()
        filtered_df = filtered_df.reset_index(level=1)
        filtered_df.index = filtered_df.index+pd.DateOffset(1)
        filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])
        
        #getting all the dates
        dates = filtered_df.index.get_level_values('date').unique().tolist()
        fixed_dates = {}
        for date in dates:
            fixed_dates[date.strftime('%Y-%m-%d')] = filtered_df.xs(date, level=0).index.tolist()
        
        stocks = aggregate_data.index.get_level_values('ticker').unique().tolist()
        
        new_df = yf.download(tickers=stocks,
                            start=aggregate_data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                            end=aggregate_data.index.get_level_values('date').unique()[-1])
        
        returns_df = np.log(new_df['Adj Close']).diff()
        portfolio_df = pd.DataFrame()
        for start in fixed_dates.keys():
            try:
                end_date = (pd.to_datetime(start)+pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
                cols = fixed_dates[start]
                optimize_start = (pd.to_datetime(start)-pd.DateOffset(months=12)).strftime('%Y-%m-%d')
                optimize_end = (pd.to_datetime(start)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                # optimize_end = (pd.to_datetime(end_date)-pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                optimization_df = new_df[optimize_start:optimize_end]['Adj Close'][cols]
                flag = False
                try:
                    weights = self.optimize_portfolio_weights(prices=optimization_df,lower_bound=round(1/(len(optimization_df.columns)*2),3))
                    weights = pd.DataFrame(weights, index=pd.Series(0))
                    flag = True
                except:
                    print(f'Max Sharpe Optimization failed for {start}, continuing with equal-eights')
                if not flag:
                    weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))], index=optimization_df.columns.tolist(),columns=pd.Series(0)).T
                # print("Columns of returns_df: ", returns_df.columns.to_list())
                # print("\n")
                # print("Columns of weights_df: ", weights.columns.to_list())
                temp_df = returns_df[start:end_date]
                temp_df = temp_df.stack().to_frame('return').reset_index(level=0)\
                    .merge(weights.stack().to_frame('weight').reset_index(level=0,drop=True),
                                                                            left_index=True,right_index=True)\
                                                                                .reset_index().set_index(['Date', 'index']).unstack().stack()
                if len(temp_df.columns.tolist()) == 0:
                    continue
                # print("\n")
                # print("Temp columns: temp_df.columns.tolist()")
                temp_df.index.names = ['date', 'ticker']
                temp_df['weighted_return'] = temp_df['return']*temp_df['weight']
                temp_df = temp_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
                portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
            except Exception as e:
                print("Exception is: " + e)
        return portfolio_df
