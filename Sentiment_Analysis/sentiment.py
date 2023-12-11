import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import *
import yfinance as yf
plt.style.use('ggplot')

class Sentiment_Model():
    def __init__(self, sentiment_df):
        self.sentiment_df = sentiment_df
        
    def filter_by_metric(self, metric: str):
        aggregate_data = (self.sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='M'), 'symbol']))[[metric]].mean()
        aggregate_data['rank'] = (aggregate_data.groupby(level=0)[metric]
        .transform(lambda x: x.rank(ascending=False)))
        return aggregate_data
    
    def analyze(self):
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])
        self.sentiment_df = self.sentiment_df.set_index(['date', 'symbol'])
        self.sentiment_df['engagement_ratio'] = self.sentiment_df['twitterComments']/self.sentiment_df['twitterLikes']
        self.sentiment_df = self.sentiment_df[(self.sentiment_df['twitterLikes']>20)&(self.sentiment_df['twitterComments']>10)]
        aggregate_data = self.filter_by_metric('twitterComments')
        
        filtered_df = aggregate_data[aggregate_data['rank']<6].copy()
        filtered_df = filtered_df.reset_index(level=1)
        filtered_df.index = filtered_df.index+pd.DateOffset(1)
        filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])
        filtered_df.head(20)
        
        dates = filtered_df.index.get_level_values('date').unique().tolist()
        fixed_dates = {}
        print("Starting download of price tickers")
        for start in dates:
            fixed_dates[start.strftime('%Y-%m-%d')] = filtered_df.xs(start, level=0).index.tolist()
            stock_list = self.sentiment_df.index.get_level_values('symbol').unique().tolist()
            prices_df = yf.download(tickers=stock_list,start='2021-01-01',end='2023-03-01')
            returns_df = np.log(prices_df['Adj Close']).diff().dropna()
        print("Finished downloading price tickers")
        return returns_df, fixed_dates
        