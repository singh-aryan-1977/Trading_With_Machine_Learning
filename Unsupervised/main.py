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
import warnings

sp500_df = pd.DataFrame()

# Calculating garman-klass volatility
def calculate_gk_volatility():
    global sp500_df
    if sp500_df.empty:
        print("Reached here 2")
        sp500_df = get_SP500()
    gk_vol_first_part = ((np.log(sp500_df['high'])-np.log(sp500_df['low'])) ** 2) // 2
    gk_vol_second_part = (2*np.log(2)-1)*(np.log(sp500_df['adj close']) - np.log(sp500_df['open']))**2
    sp500_df['garman_klass_volatility'] = gk_vol_first_part - gk_vol_second_part

def main():
    global sp500_df
    if sp500_df.empty:
        print("Reached here")
        sp500_df = get_SP500()
    calculate_gk_volatility()
    # print(sp500_df.columns.tolist())
    print(sp500_df)
    
if __name__ == "__main__":
    main()