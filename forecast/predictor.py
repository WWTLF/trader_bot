import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import backtesting
from backtesting import Backtest, Strategy
from utils.simple_strategy import SimpleFollowSignalsStrategy
from sklearn.preprocessing import MinMaxScaler

class Predictor():
    def __init__(self):
        self.traindedTechParams = {}

    
    def predict(self, ticker: str, df: pd.DataFrame) -> str:
        pass





    def calculate_signal_return(self, signal_df: pd.DataFrame, params: dict) -> tuple[float, str]:


        test_df = signal_df.copy()
        test_df['Signal'] = 0

        test_df['final_weight'] = params['bb_weight'] * test_df['bb_signal_weight'] 
        + params['sma_weight'] * test_df['sma_signal_weight'] 
        + params['rsi_weight'] * test_df['rsi_signal_weight'] 
        + params['macd_weight'] * test_df['macd_signal_weight'] 
        # + test_df['ema_crossing_weight'] * params['ema_crossing_weight']

        test_df.loc[test_df['final_weight'] > params['buy_th'], 'Signal'] = 1
        test_df.loc[test_df['final_weight'] < params['sell_th'], 'Signal'] = -1

        test_df['Open'] = test_df['open']
        test_df['Close'] = test_df['close']
        test_df['High'] = test_df['high']
        test_df['Low'] = test_df['low']
        test_df['volume'] = test_df['volume']

        bt = Backtest(test_df, SimpleFollowSignalsStrategy, cash=2*test_df.iloc[0]['close'], commission=.002, exclusive_orders=True)
        stats = bt.run()
        last_signal = test_df.iloc[-1]['Signal']

        return stats['Return [%]'], (1 if last_signal == 1 else -1 if last_signal == -1 else 0)