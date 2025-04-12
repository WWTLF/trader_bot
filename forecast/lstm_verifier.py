# import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import datetime

from repositories.model_config import MLModelConfigRepo
from sklearn.preprocessing import MinMaxScaler
from models.ml_model_config import MlModelConfig
from repositories.extra_feature import ExtraFeatureRepository
import math

class LSTMVerifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout_rate=0.3, l1_lambda=0.00001, l2_lambda=0.0002):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Берём только последний выход
        out = self.fc(out)
        return self.sigmoid(out)
    
    def regularization_loss(self):
        l1_loss = sum(torch.abs(param).sum() for param in self.parameters())
        l2_loss = sum(torch.square(param).sum() for param in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


class LSTMVerifierService:
    def __init__(self, ticker, db_connection):
        self.signals_df = None
        self.ticker = ticker
        self.db_connection = db_connection
        self.model_config = None
        self.features = []



    def get_mdoel_config(self) -> MlModelConfig:
        return self.model_config

    def deserizalize(self, signals_df: pd.DataFrame, initial_price: float = 0.0):
        model_config_repo = MLModelConfigRepo(conn=self.db_connection)
        self.model_config = model_config_repo.get_by_name("LSTM_VERIFIER_{}".format(self.ticker))
        self.features = self.model_config.config['features']
        self.prev_filtered_price = initial_price
        self.signals_df = signals_df



    def predict(self, date: datetime.datetime) -> float:

        extra_features_repo = ExtraFeatureRepository(conn=self.db_connection)


        cpu = torch.device("cpu")
        sequence_length = self.model_config.config['seq_length']
        pred_df = self.signals_df.loc[self.signals_df.index <= date]

        scalerX = MinMaxScaler()
        scalerX.min_ = self.model_config.config['scalerMinX']
        scalerX.scale_ =  self.model_config.config['scalerSlaceX']

        pred_df = pred_df[-sequence_length:]
        extra_features = extra_features_repo.get_all_for_ticker_and_date(self.ticker, pred_df.index[0], pred_df.index[-1])
        pred_df['pct_real_close'] = 0.0
        pred_df['pct_pred_close'] = 0.0


        for idx, row in pred_df.iterrows():
            if idx in extra_features.index:
                # pred_df.loc[idx, 'pct_real_close'] = extra_features.loc[idx]['pct_real_close']
                pred_df.loc[idx, 'pct_pred_close'] = extra_features.loc[idx]['pct_pred_close']
                # pred_df.loc[idx, 'pct_pred_close_real_close'] = extra_features.loc[idx]['pct_pred_close_real_close']
       

        pred_df['pct_pred_close_real_close'] =  pred_df['pct_pred_close'] - pred_df['pct_real_close']

        X = scalerX.transform(pred_df[self.features])
        sequences = []
        sequences.append(X)
        X_valid = torch.tensor(np.array(sequences), dtype=torch.float32)

        model = LSTMVerifier(input_size  = self.model_config.config['input_size'], 
                              hidden_size = self.model_config.config['hidden_size'], 
                              num_layers  = self.model_config.config['num_layers'], 
                              l1_lambda=0.0, l2_lambda=0.0001, dropout_rate=0.3)


        model.load_state_dict(torch.load(self.model_config.path))

        model.eval()
        model.to(cpu)
        with torch.no_grad():
            valid_predictions = model(X_valid) # Предсказания на валидационном DS


        non_trust_level = valid_predictions[0].item()
        # print(pred_price)

        if math.isnan(non_trust_level):
            return 1.0

        return non_trust_level

