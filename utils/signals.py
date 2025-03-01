import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelextrema

def add_signals(ticker_df: pd.DataFrame):
        ticker_df['RSI_signal'] = 0
        ticker_df.loc[(ticker_df['RSI'] < 30), 'RSI_signal'] = 1
        ticker_df.loc[(ticker_df['RSI'] > 70), 'RSI_signal'] = -1
        ticker_df["EMA_Cross_signal"] = np.where(ticker_df["EMA_9"] > ticker_df["EMA_21"], 1, -1)  # 1 - бычий тренд, -1 - медвежий
        ticker_df["macd_signal"] = get_filtered_macd_signals(ticker_df)
        ticker_df["sma_signal"] = np.where(ticker_df["SMA_50"] > ticker_df["SMA_200"], 1, np.where(ticker_df["SMA_50"] < ticker_df["SMA_200"], -1, 0))
        ticker_df["bb_signal"] = np.where(ticker_df["close"] < ticker_df["bb_lower"], 1, np.where(ticker_df["close"] > ticker_df["bb_upper"], -1, 0))
        ticker_df['OBV_Signal'] = np.where(ticker_df['OBV'].diff() > 0, 1, -1)
        ticker_df['AD_Signal'] = np.where(ticker_df['AD'].diff() > 0, 1, -1)
        ticker_df['MFI_Signal'] = np.where(ticker_df['MFI'] < 20, 1, np.where(ticker_df['MFI'] > 80, -1, 0))
        ticker_df['VO_Signal'] = np.where(ticker_df['VO'] > 0, 1, -1)
        ticker_df['Stochastic_Signal'] = np.where(ticker_df['Stochastic_K'] < 20, 1, np.where(ticker_df['Stochastic_K'] > 80, -1, 0))
        ticker_df['CCI_Signal'] = np.where(ticker_df['CCI'] < -100, 1, np.where(ticker_df['CCI'] > 100, -1, 0))

        ticker_df['close_to_volume'] = ticker_df['close'] / ticker_df['volume']
        ticker_df['close_by_volume'] = ticker_df['close'] * ticker_df['volume']
        ticker_df['sma_signal_2'] = np.where(ticker_df['SMA_10'] > ticker_df['SMA_20'], 1, -1)
        mark_trade_signals(ticker_df, 'train_signal_command_4')

def get_filtered_macd_signals(df):
    """Функция для генерации MACD сигналов с фильтром SMA"""
    signals = []
    
    for i in range(len(df)):
        if i < 1:  # Пропускаем первый индекс (нет данных для сравнения)
            signals.append(0)
            continue
        
        macd_prev, macd_now = df["macd"].iloc[i-1], df["macd"].iloc[i]
        signal_prev, signal_now = df["macd_signal_line"].iloc[i-1], df["macd_signal_line"].iloc[i]
        close_now, ma_now = df["close"].iloc[i], df["SMA_200"].iloc[i]
        
        if macd_prev < signal_prev and macd_now > signal_now and close_now > ma_now:
            signals.append(1)  # Покупка
        elif macd_prev > signal_prev and macd_now < signal_now and close_now < ma_now:
            signals.append(-1)  # Продажа
        else:
            signals.append(0)  # Держим

    return signals


def add_signal_weight(signal_df: pd.DataFrame, signal_column_name: str ,singal_weight_column: str):
    prev_sig_weight = 0.0
    signal_df[singal_weight_column] = 0.0
    for index, row in signal_df.iterrows():      
        if row[signal_column_name] == 1:         
            signal_df.loc[index, singal_weight_column] = prev_sig_weight + 1              
        elif row[signal_column_name] == -1:        
            signal_df.loc[index, singal_weight_column] = prev_sig_weight - 1   
        else:
            signal_df.loc[index, singal_weight_column] = prev_sig_weight / 2.0
        prev_sig_weight = signal_df.loc[index, singal_weight_column]
    # Нормализуем вес к 1
    scaler = MinMaxScaler()
    signal_df[singal_weight_column]  = scaler.fit_transform(signal_df[[singal_weight_column]]) - 0.5




def mark_trade_signals(df, column_name):
    df[column_name] = 0  # По умолчанию нет сигнала
    
    # Ищем локальные минимумы (покупки)
    local_min = argrelextrema(df['close'].values, np.less, order=5)[0]
    df.loc[df.index[local_min], column_name] = 1  # Покупка
    
    # Ищем локальные максимумы (продажи)
    local_max = argrelextrema(df['close'].values, np.greater, order=5)[0]
    df.loc[df.index[local_max], column_name] = -1  # Продажа
