from download import *
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
from indicators import IndicatorCalculator
from unsupervised import UnsupervisedModel


def plot_strategy_returns(stock, portfolio_df):
     # Download returns for SP500
    spy = yf.download(tickers=stock,start='2015-01-01',end=dt.date.today())
    spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close':f'{stock} Buy&Hold'},axis=1)
    portfolio_df = portfolio_df.merge(spy_ret, left_index=True,right_index=True)
    
    plt.style.use('ggplot')
    portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1
    portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))
    plt.show()

def main():
    sp500_df = get_SP500()
    indicator_calculator = IndicatorCalculator(sp500_df=sp500_df)
    sp500_df = indicator_calculator.calculate_simple_indicators(sp500_df=sp500_df)
    unsupervised_model = UnsupervisedModel(sp500_df=sp500_df)
    portfolio_df = unsupervised_model.learn()
    plot_strategy_returns('MSFT', portfolio_df=portfolio_df)

if __name__ == "__main__":
    main()