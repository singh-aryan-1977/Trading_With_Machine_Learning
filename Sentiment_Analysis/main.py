import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import *
import yfinance as yf
import os
import matplotlib.ticker as mtick
from sentiment import Sentiment_Model
plt.style.use('ggplot')

    
def plot(ticker, portfolio_df, factor):
    print("Downloading NASDAQ tickeres")
    qqq_df = yf.download(tickers=ticker,start='2021-01-01',end='2023-03-01')
    qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')
    portfolio_df = portfolio_df.merge(qqq_ret,left_index=True,right_index=True)
    portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)
    portfolios_cumulative_return.plot(figsize=(16,6))
    plt.title(f'{factor} Strategy Return Over Time')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Return')
    plt.show()
    return

def main():
    file_path = 'sentiment_data.csv'  

    if os.path.exists(file_path): 
        print("Downloading Sentiment Data")
        sentiment_df = pd.read_csv(file_path)
    else:
        print(f"The file '{file_path}' does not exist.")
        return
    factor='engagementRatio'
    sentiment_model = Sentiment_Model(sentiment_df=sentiment_df)
    returns_df, fixed_dates = sentiment_model.analyze(factor=factor)
    portfolio_df = pd.DataFrame()
    for start in fixed_dates.keys():
        end = (pd.to_datetime(start)+pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
        cols = fixed_dates[start]
        temp_df = returns_df[start:end][cols].mean(axis=1).to_frame('portfolio_return')
        portfolio_df = pd.concat([portfolio_df, temp_df],axis=0)
    plot(ticker='QQQ', portfolio_df=portfolio_df, factor=factor)
    return
    
    
if __name__ == "__main__":
    main()