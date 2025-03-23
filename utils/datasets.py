from utils.download import download_all
from utils.download import get_all_stock
from utils.features import add_features
from utils.signals import add_signals
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

def get_dataset_for_ticker(signals_df, features: list, target_price_column: str, sequence_length: int, offset: int, training_dataset_size_months: int, validation_last_months: int, adjust_scalers: float = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame):
        # Возьмем для примера акции GOOGLE и на них будим исследовать наши модели

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    # target_price_column = 'close'

    data = signals_df.copy()


    last_day = data.index[-1] # Последнм днем считаем, последний загруженный в базу данных день

    device = torch.device("cuda:0")
    cpu = torch.device("cpu")

    train_df = data.loc[(last_day - relativedelta(months=validation_last_months + training_dataset_size_months) < data.index ) & (data.index < last_day - relativedelta(months=validation_last_months))]
    # На обучении модель никогда не увидит эти данных, чтобы исключить проблему заглядывания в будущее
    valid_df = data.loc[(data.index >  last_day - relativedelta(months=validation_last_months, days=sequence_length) )]
    train_size = int(len(train_df) * 0.95)


    print(len(train_df))
    sequences = []
    labels = []

    # # Создаем два скейлера и настраиваем их строго на тренировочные данные
    scalerX.fit(train_df[:train_size][features])
    scalerY.fit(train_df[:train_size][[target_price_column]])

    learn_data = scalerX.transform(train_df[:train_size][features])
    learn_target_prices = scalerY.transform(train_df[:train_size][[target_price_column]])

    for i in range(sequence_length, len(learn_data)-offset):
        sequences.append(learn_data[i-sequence_length:i])
        labels.append(learn_target_prices[i+offset])

    X = np.array(sequences)
    y = np.array(labels)
    # Преобразование данных в PyTorch tensors
    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)

    sequences = []
    labels = []

    if adjust_scalers:
        scalerX2 = MinMaxScaler()
        scalerY2 = MinMaxScaler()
        scalerX2.fit(train_df[features])
        scalerY2.fit(train_df[[target_price_column]])
        scalerX = scalerX2
        scalerY = scalerY2

    test_data = scalerX.transform(train_df[train_size-sequence_length:][features])
    test_targets = scalerY.transform(train_df[train_size-sequence_length:][[target_price_column]])
    for i in range(sequence_length, len(test_data)-offset):
        sequences.append(test_data[i-sequence_length:i])
        labels.append(test_targets[i+offset])

    X = np.array(sequences)
    y = np.array(labels)
    X_test = torch.tensor(X, dtype=torch.float32).to(device)
    y_test = torch.tensor(y, dtype=torch.float32).to(device)


    sequences = []
    labels = []

    if adjust_scalers:
        scalerX2 = MinMaxScaler()
        scalerY2 = MinMaxScaler()
        scalerX2.fit(data.loc[(last_day - relativedelta(months=training_dataset_size_months) < data.index)][features])
        scalerY2.fit(data.loc[(last_day - relativedelta(months=training_dataset_size_months) < data.index)][[target_price_column]])
        scalerX = scalerX2
        scalerY = scalerY2

    valid_data = scalerX.transform(valid_df[features])
    valid_targets = scalerY.transform(valid_df[[target_price_column]])
    for i in range(sequence_length, len(valid_data)-offset):
        sequences.append(valid_data[i-sequence_length:i])
        labels.append(valid_targets[i+offset])

    X = np.array(sequences)
    y = np.array(labels)
    X_valid = torch.tensor(X, dtype=torch.float32)
    y_valid = torch.tensor(y, dtype=torch.float32)

    return X_test, y_test, X_train, y_train, X_valid, y_valid, valid_df, scalerX.min_, scalerX.scale_, scalerY.min_, scalerY.scale_