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

class TSMixerBlock(nn.Module):
    def __init__(self, seq_length, feature_dim, expansion_factor=4, dropout=0.1):
        """
        A single TSMixer block that consists of two MLP layers: 
        - One mixing across the time dimension.
        - Another mixing across the feature dimension.
        
        Args:
            sequence_length (int): Number of past time steps.
            feature_dim (int): Number of input features.
            expansion_factor (int): Expansion factor for the hidden layer in MLP.
            dropout (float): Dropout rate for regularization.
        """
        super(TSMixerBlock, self).__init__()

        self.time_mixer = nn.Sequential(
            nn.Linear(seq_length, seq_length * expansion_factor),
            nn.GELU(),
            nn.Linear(seq_length * expansion_factor, seq_length),
            nn.Dropout(dropout)
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(feature_dim * expansion_factor, feature_dim),
            nn.Dropout(dropout)
        )

        self.norm_t = nn.LayerNorm([seq_length, feature_dim])
        self.norm_f = nn.LayerNorm([seq_length, feature_dim])

    def forward(self, x):
        """
        Forward pass of TSMixerBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
            torch.Tensor: Transformed tensor of the same shape.
        """
        # Residual connection for time mixing
        x = x + self.time_mixer(self.norm_t(x).transpose(1, 2)).transpose(1, 2)

        # Residual connection for feature mixing
        x = x + self.feature_mixer(self.norm_f(x))

        return x


class TSMixer(nn.Module):
    def __init__(self, seq_length, feature_dim, num_layers=4, expansion_factor=4, dropout=0.1,  l1_lambda=0.00001, l2_lambda=0.0001):
        """
        TSMixer model, which stacks multiple TSMixer blocks.
        
        Args:
            sequence_length (int): Number of past time steps.
            feature_dim (int): Number of input features.
            num_blocks (int): Number of stacked TSMixer blocks.
            expansion_factor (int): Expansion factor for the hidden layers in MLPs.
            dropout (float): Dropout rate for regularization.
        """
        super(TSMixer, self).__init__()
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_length, feature_dim, expansion_factor, dropout) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(feature_dim, 1)  # Predicting single value per timestep
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        """
        Forward pass of TSMixer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, 1).
        """
        for block in self.blocks:
            x = block(x)
        x = x[:, -1, :]  # Shape: (batch_size, feature_dim)
        return self.output_layer(x)  # Predict target feature
    

    def regularization_loss(self):
        l1_loss = sum(torch.abs(param).sum() for param in self.parameters())
        l2_loss = sum(torch.square(param).sum() for param in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


class TSMixerPredictorService:
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
        self.model_config = model_config_repo.get_by_name("TSMixer_{}".format(self.ticker))
        self.features = self.model_config.config['features']
        self.prev_filtered_price = initial_price
        self.signals_df = signals_df

        self.kf = KalmanFilter(
            transition_matrices=[1],  # Цена меняется плавно
            observation_matrices=[1],  # Мы наблюдаем цену напрямую
            initial_state_mean=initial_price,  # Первая цена (например, средняя)
            initial_state_covariance=0.001,  # Начальная неопределенность (уменьшена)
            observation_covariance=0.001,  # Ошибка предсказания LSTM (чем меньше, тем сильнее сглаживание)
            transition_covariance=0.0001  # Насколько плавно менять (чем меньше, тем более плавный тренд)
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


        model = TSMixer(seq_length=self.model_config.config['seq_length'], 
                num_layers=self.model_config.config['num_layers'], 
                feature_dim=self.model_config.config['feature_dim'], 
                expansion_factor=self.model_config.config['expansion_factor'],
                dropout=0.2, l1_lambda=0.0, l2_lambda=0.001)
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

