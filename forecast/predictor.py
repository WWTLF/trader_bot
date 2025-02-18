# import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from utils.simple_strategy import SimpleFollowSignalsStrategy
# from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np


class ActionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3,  l1_lambda=0.0, l2_lambda=0.0001):
        super(ActionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)
        # self.soft_max = nn.Softmax(dim=1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
       
        # out = self.soft_max(out)
        
        return out
    
    def regularization_loss(self):
        l1_loss = sum(torch.abs(param).sum() for param in self.parameters())
        l2_loss = sum(torch.square(param).sum() for param in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

class Predictor():
    def __init__(self, features: list, sequence_length=32):
        self.traindedTechParams = {}
        self.sequence_length = sequence_length

            # Гиперпараметры
        input_size = len(features)
        hidden_size = 64
        num_layers = 1
        output_size = 3  # 3 класса: Buy, Sell, Hold
        self.model = ActionLSTMModel(input_size, hidden_size, num_layers, output_size)
        self.model.load_state_dict(torch.load('./models/action_forcast_lstm_model.pth'))
    
    def predict(self, df: pd.DataFrame) -> int:  
        sequences = []
        sequences.append(df)
        X = np.array(sequences, dtype=np.float32)
        valid_prediction = self.model(X)
        predicted_sginal = torch.argmax(valid_prediction, dim=1)[0] - 1
        return predicted_sginal




    # def calculate_signal_return(self, signal_df: pd.DataFrame, params: dict) -> tuple[float, str]:


    #     test_df = signal_df.copy()
    #     test_df['Signal'] = 0

    #     test_df['final_weight'] = params['bb_weight'] * test_df['bb_signal_weight'] 
    #     + params['sma_weight'] * test_df['sma_signal_weight'] 
    #     + params['rsi_weight'] * test_df['rsi_signal_weight'] 
    #     + params['macd_weight'] * test_df['macd_signal_weight'] 
    #     # + test_df['ema_crossing_weight'] * params['ema_crossing_weight']

    #     test_df.loc[test_df['final_weight'] > params['buy_th'], 'Signal'] = 1
    #     test_df.loc[test_df['final_weight'] < params['sell_th'], 'Signal'] = -1

    #     test_df['Open'] = test_df['open']
    #     test_df['Close'] = test_df['close']
    #     test_df['High'] = test_df['high']
    #     test_df['Low'] = test_df['low']
    #     test_df['volume'] = test_df['volume']

    #     bt = Backtest(test_df, SimpleFollowSignalsStrategy, cash=2*test_df.iloc[0]['close'], commission=.002, exclusive_orders=True)
    #     stats = bt.run()
    #     last_signal = test_df.iloc[-1]['Signal']

    #     return stats['Return [%]'], (1 if last_signal == 1 else -1 if last_signal == -1 else 0)