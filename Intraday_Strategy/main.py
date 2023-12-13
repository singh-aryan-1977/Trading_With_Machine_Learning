import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
from intraday import Intraday_Model
import matplotlib.ticker as mtick
import numpy as np

def plot(strategy_cumulative_return):
    strategy_cumulative_return.plot(figsize=(16,6))
    plt.title('Intraday Strategy Returns')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel('Returns')
    plt.show()
    return

def main():
    daily_df = pd.read_csv("simulated_daily_data.csv")
    # print(daily_df)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.set_index('Date')
    daily_df['log_ret'] = np.log(daily_df['Adj Close']).diff()
    five_min_df = pd.read_csv("simulated_5min_data.csv")
    five_min_df['datetime'] = pd.to_datetime(five_min_df['datetime'])
    five_min_df = five_min_df.set_index('datetime')
    five_min_df['date'] = five_min_df.index.date
    five_min_df['date'] = pd.to_datetime(five_min_df['date'])
    # print(five_min_df)
    # print(five_min_df.columns.tolist())
    # print(five_min_df)
    itra_model = Intraday_Model(daily_df=daily_df, five_min_df=five_min_df)
    daily_return_df = itra_model.return_final_df()
    strategy_cumulative_return = np.exp(np.log1p(daily_return_df).cumsum()).sub(1)
    plot(strategy_cumulative_return=strategy_cumulative_return)
    return
if __name__ == "__main__":
    main()