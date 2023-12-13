
from arch import arch_model
import pandas as pd
import pandas_ta
import numpy as np

class Intraday_Model():
    def __init__(self, daily_df, five_min_df):
        self.daily_df = daily_df
        self.five_min_df = five_min_df
        
    def predict_volatility(self,x,p,q):
        best_model = arch_model(y=x*100,p=p,q=q).fit(update_freq=5, disp='off')
        variance_forecast = best_model.forecast(horizon=1).variance.iloc[-1,0]    
        return variance_forecast
    
    def calculating_daily_signal(self):
        self.daily_df['variance'] = self.daily_df['log_ret'].rolling(180).var()
        self.daily_df['predictions'] = self.daily_df['log_ret'].rolling(180).apply(lambda x: self.predict_volatility(x,2,3))
        self.daily_df['prediction_premium'] = (self.daily_df['predictions']-self.daily_df['variance'])/self.daily_df['variance']
        self.daily_df['premium_std'] = self.daily_df['prediction_premium'].rolling(180).std()
        self.daily_df['signal_daily'] = self.daily_df.apply(lambda x: 1 if (x['prediction_premium']>x['premium_std']*1.5)
                                                else (-1 if (x['prediction_premium']<x['premium_std']*-1.5) else np.nan),axis=1)
        return self.daily_df
        
    
    def get_indicators(self, daily_df_2):
        daily_df_2['signal_daily'] = daily_df_2['signal_daily'].shift()
        final_df = self.five_min_df.reset_index().merge(daily_df_2[['signal_daily']].reset_index(),left_on='date',right_on='Date').set_index('datetime')
        final_df['rsi'] = pandas_ta.rsi(close=final_df['close'],length=20)
        final_df['lband'] = pandas_ta.bbands(close=final_df['close'],length=20).iloc[:,0]
        final_df['uband'] = pandas_ta.bbands(close=final_df['close'],length=20).iloc[:,2]
        final_df['signal_intraday'] = final_df.apply(lambda x: 1 if (x['rsi']>70)&(x['close']>x['uband'])
                                                    else (-1 if (x['rsi']<70)&(x['close']<x['lband']) else np.nan)
                                                    ,axis=1)
        final_df['return_sign'] = final_df.apply(lambda x: -1 if (x['signal_daily']==1)&(x['signal_intraday']==1)
                                             else (1 if (x['signal_daily']==-1)&(x['signal_intraday']==-1) else np.nan),
                                                   axis=1)
        return final_df
        
    def return_final_df(self):
        self.daily_df = self.calculating_daily_signal()
        final_df = self.get_indicators(self.daily_df)
        final_df['return_sign'] = final_df.groupby(pd.Grouper(freq='D'))['return_sign'].transform(lambda x: x.ffill())
        final_df['return'] = final_df['close'].pct_change()
        final_df['forward_return'] = final_df['return'].shift(-1)
        final_df['strategy_return'] = final_df['forward_return']*final_df['return_sign']
        daily_return_df = final_df.groupby(pd.Grouper(freq='D'))[['strategy_return']].sum()
        return daily_return_df