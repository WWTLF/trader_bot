import pandas as pd
import numpy as np

def add_signals(ticker_df: pd.DataFrame):
        ticker_df['RSI_signal'] = 0
        ticker_df.loc[(ticker_df['RSI'] < 30), 'RSI_signal'] = 1
        ticker_df.loc[(ticker_df['RSI'] > 70), 'RSI_signal'] = -1
        ticker_df["EMA_Cross_signal"] = np.where(ticker_df["EMA_9"] > ticker_df["EMA_21"], 1, -1)  # 1 - бычий тренд, -1 - медвежий
        ticker_df["macd_signal"] = get_filtered_macd_signals(ticker_df)





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