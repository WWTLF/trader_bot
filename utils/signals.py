import pandas as pd
import numpy as np

def add_signals(ticker_df: pd.DataFrame):
        ticker_df['RSI_signal'] = 'hold'
        ticker_df.loc[(ticker_df['RSI'] < 30), 'RSI_signal'] = 'buy'
        ticker_df.loc[(ticker_df['RSI'] > 70), 'RSI_signal'] = 'sell'
        ticker_df["EMA_Cross_signal"] = np.where(ticker_df["EMA_9"] > ticker_df["EMA_21"], 'buy', 'sell')  # 1 - бычий тренд, -1 - медвежий
        ticker_df["macd_signal"] = get_filtered_macd_signals(ticker_df)
        ticker_df["bb_signal"] = np.where(ticker_df["close"] < ticker_df["bb_lower"], "buy", np.where(ticker_df["ticker_df"] > ticker_df["bb_uper"], "sell", "hold"))





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
            signals.append("buy")  # Покупка
        elif macd_prev > signal_prev and macd_now < signal_now and close_now < ma_now:
            signals.append("sell")  # Продажа
        else:
            signals.append("hold")  # Держим

    return signals



def add_signal_weight(signal_df: pd.DataFrame, signal_column_name: str ,singal_weight_column: str):
    prev_sig_weight = 0.0
    signal_df[singal_weight_column] = 0.0
    for index, row in signal_df.iterrows():      
        if row[signal_column_name] == "buy":         
            signal_df.loc[index, singal_weight_column] = prev_sig_weight + 1              
        elif row[signal_column_name] == "sell":        
            signal_df.loc[index, singal_weight_column] = prev_sig_weight - 1   
        else:
            signal_df.loc[index, singal_weight_column] = prev_sig_weight / 2.0
        prev_sig_weight = signal_df.loc[index, singal_weight_column]
    # Нормализуем вес к 1
    scaler = MinMaxScaler()
    signal_df[singal_weight_column]  = scaler.fit_transform(signal_df[[singal_weight_column]]) - 0.5