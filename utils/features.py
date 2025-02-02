import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
import talib
import numpy as np

def add_features(source_df: pd.DataFrame) -> pd.DataFrame:
    # scaler = MinMaxScaler()
    for ticker in source_df['ticker'].unique():
            ticker_df = source_df[source_df['ticker'] == ticker]
            # ticker_df['scaled_close_price'] = scaler.fit_transform(ticker_df[['close']])

            ticker_df['tema'] = talib.TEMA(ticker_df['close'], timeperiod=24)
            ticker_df['macd'], ticker_df['macd_signal_line'], ticker_df['macd_hist'] = talib.MACD(ticker_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
          
            # Создаем сигналы для покупки и продажи
            # Добавляем SMA/EMA для фильтрации тренда
            ticker_df["SMA_50"] = talib.SMA(ticker_df["close"], timeperiod=50)
            ticker_df["SMA_200"] = talib.SMA(ticker_df["close"], timeperiod=200)
            ticker_df["EMA_50"] = talib.EMA(ticker_df["close"], timeperiod=50)
         

            ticker_df["EMA_9"] = talib.EMA(ticker_df["close"], timeperiod=9)
            ticker_df["EMA_21"] = talib.EMA(ticker_df["close"], timeperiod=21)        
            ticker_df['RSI'] =  talib.RSI(ticker_df["close"], timeperiod=14)


            mark_optimal_trades(ticker_df)


            # Возвращаем данные в оригинальный датафрейм по всем ценным бумагам
            # source_df.loc[source_df['ticker'] == ticker, 'scaled_close_price'] = ticker_df['scaled_close_price']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'macd_signal_line'] = ticker_df['macd_signal_line']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'tema'] = ticker_df['tema']
            source_df.loc[source_df['ticker'] == ticker, 'RSI'] = ticker_df['RSI']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_50'] = ticker_df['EMA_50']
            source_df.loc[source_df['ticker'] == ticker, 'SMA_50'] = ticker_df['SMA_50']
            source_df.loc[source_df['ticker'] == ticker, 'optimal_signal'] = ticker_df['optimal_signal']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_9'] = ticker_df['EMA_9']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_21'] = ticker_df['EMA_21']


    source_df['daily_return'] = source_df.groupby('ticker')['close'].pct_change()
    return source_df[1:]



def mark_optimal_trades(df, future_window=10, price_change_threshold=0.10):
    """Размечает идеальные сигналы Buy (1), Sell (-1), Hold (0), используя будущее"""
    
    # df = df.copy()
    signals = np.zeros(len(df))  # По умолчанию все сигналы Hold (0)

    for i in range(len(df) - future_window):
        current_price = df["close"].iloc[i]
        
        # Будущие цены в окне future_window
        future_prices = df["close"].iloc[i+1:i+1+future_window]

        # Будущие максимумы и минимумы
        future_max = future_prices.max()
        future_min = future_prices.min()

        # Оптимальная покупка (если цена сильно вырастет в будущем)
        if (future_max - current_price) / current_price >= price_change_threshold:
            signals[i] = 1  # Buy

        # Оптимальная продажа (если цена сильно упадет в будущем)
        elif (current_price - future_min) / current_price >= price_change_threshold:
            signals[i] = -1  # Sell

    df["optimal_signal"] = signals
    return df