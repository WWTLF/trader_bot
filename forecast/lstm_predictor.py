# import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import datetime

from repositories.model_config import MLModelConfigRepo
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from models.ml_model_config import MlModelConfig

class WeekPriceForcastLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.1, l1_lambda=0.0, l2_lambda=0.0001):
        super(WeekPriceForcastLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc1_activation = nn.LeakyReLU()

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])
        out = self.fc1_activation(out)

        # out = self.fc2_activation(out)
        # out = self.fc3(out)
        return out
    
    def regularization_loss(self):
        l1_loss = sum(torch.abs(param).sum() for param in self.parameters())
        l2_loss = sum(torch.square(param).sum() for param in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


class LSTMPredictorService:
    def __init__(self, ticker, db_connection):
        self.signals_df = None
        self.ticker = ticker
        self.db_connection = db_connection
        self.model_config = None
        self.features = []
        self.kf = None
        self.prev_filtered_price = 0.0
        self.filtered_covariance = 0.01 



    def get_mdoel_config(self) -> MlModelConfig:
        return self.model_config

    def deserizalize(self, signals_df: pd.DataFrame, initial_price: float = 0.0):
        model_config_repo = MLModelConfigRepo(conn=self.db_connection)
        self.model_config = model_config_repo.get_by_name("LSTM_{}".format(self.ticker))
        self.features = self.model_config.config['features']
        self.prev_filtered_price = initial_price
        self.signals_df = signals_df

        self.kf = KalmanFilter(
            transition_matrices=[1],  # Цена меняется плавно
            observation_matrices=[1],  # Мы наблюдаем цену напрямую
            initial_state_mean=initial_price,  # Первая цена (например, средняя)
            initial_state_covariance=0.01,  # Начальная неопределенность (уменьшена)
            observation_covariance=0.005,  # Ошибка предсказания LSTM (чем меньше, тем сильнее сглаживание)
            transition_covariance=0.001  # Насколько плавно менять (чем меньше, тем более плавный тренд)
        )


    def predict(self, date: datetime.datetime) -> tuple:
        cpu = torch.device("cpu")
        sequence_length = self.model_config.config['seq_length']
        pred_df = self.signals_df.loc[self.signals_df.index <= date]

        scalerX = MinMaxScaler()
        scalerX.min_ = self.model_config.config['scalerMinX']
        scalerX.scale_ =  self.model_config.config['scalerSlaceX']

        pred_df = pred_df[-sequence_length:]
        X = scalerX.transform(pred_df[self.features])
        sequences = []
        sequences.append(X)
        X_valid = torch.tensor(np.array(sequences), dtype=torch.float32)


        model = WeekPriceForcastLSTMModel(self.model_config.config['input_size'], 
                                          self.model_config.config['hidden_size'], 
                                          self.model_config.config['num_layers'], 
                                          self.model_config.config['output_size'], l1_lambda=0.0, l2_lambda=0.0001, dropout_rate=0.3)
        model.load_state_dict(torch.load(self.model_config.path))

        model.eval()
        model.to(cpu)
        with torch.no_grad():
            valid_predictions = model(X_valid) # Предсказания на валидационном DS


        pred_price = valid_predictions[0,0].item()
        # print(pred_price)


        filtered_price, self.filtered_covariance = self.kf.filter_update(
            self.prev_filtered_price,  # Последнее отфильтрованное значение
             self.filtered_covariance,  # Ковариация
            observation=pred_price  # Новое предсказание LSTM
        )

        self.prev_filtered_price = filtered_price

        return pred_price, filtered_price[0,0].item()

