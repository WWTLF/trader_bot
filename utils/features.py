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

            ticker_df['mean_close'] = ticker_df['close'].rolling(window=7, center=True).mean()


            
            ticker_df['signal_command'] = None
            add_optimal_signals(ticker_df)

            # Возвращаем данные в оригинальный датафрейм по всем ценным бумагам
            # source_df.loc[source_df['ticker'] == ticker, 'scaled_close_price'] = ticker_df['scaled_close_price']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'macd_signal_line'] = ticker_df['macd_signal_line']
            source_df.loc[source_df['ticker'] == ticker, 'macd'] = ticker_df['macd']
            source_df.loc[source_df['ticker'] == ticker, 'tema'] = ticker_df['tema']
            source_df.loc[source_df['ticker'] == ticker, 'RSI'] = ticker_df['RSI']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_50'] = ticker_df['EMA_50']
            source_df.loc[source_df['ticker'] == ticker, 'SMA_50'] = ticker_df['SMA_50']
            source_df.loc[source_df['ticker'] == ticker, 'signal_command'] = ticker_df['signal_command']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_9'] = ticker_df['EMA_9']
            source_df.loc[source_df['ticker'] == ticker, 'EMA_21'] = ticker_df['EMA_21']
            source_df.loc[source_df['ticker'] == ticker, 'mean_close'] = ticker_df['mean_close']



    source_df['daily_return'] = source_df.groupby('ticker')['close'].pct_change()
    return source_df[1:]


def add_optimal_signals(df, price_column = 'mean_close', threshold=0.05):

    df['signal_command'] = 'hold'

    current_position = 0  # 0=нет позиции, 1=лонг, -1=шорт


    # last_deal_price = df.iloc[0][price_column]
    last_deal_price = 0.001
    prev_command = 'hold'
    for i in range(1, len(df)):
        # Используем iloc, чтобы брать i-ю строку вне зависимости от значения индекса
        price_now = df.iloc[i][price_column]
        price_prev = df.iloc[i-1][price_column]

        # Если предыдущая цена ноль — пропускаем во избежание деления на 0
        if price_prev == 0:
            continue

        # Рассчитываем процентное изменение
        pct_change = (price_now - last_deal_price) / last_deal_price 

        # abs(price_now - last_deal_price)/last_deal_price > threshold2 and price_now > last_deal_price:

        if pct_change > threshold:
            if current_position == 0:
                df.iloc[i, df.columns.get_loc('signal_command')] = 'buy'
                current_position = 1
            elif current_position == -1:
                df.iloc[i, df.columns.get_loc('signal_command')] = 'cover/buy'
                current_position = 1

            if prev_command == 'cover/buy' and price_now > last_deal_price: 
                df.iloc[i, df.columns.get_loc('signal_command')] = 'hold'
                current_position == 0
            else:
                last_deal_price = price_now

            prev_command = df.iloc[i, df.columns.get_loc('signal_command')]

        elif pct_change < -threshold:
            if current_position == 0:
                df.iloc[i, df.columns.get_loc('signal_command')] = 'sell'
                current_position = -1
            elif current_position == 1:
                df.iloc[i, df.columns.get_loc('signal_command')] = 'sell/short'
                current_position = -1
            if prev_command == 'sell/short' and price_now < last_deal_price: 
                df.iloc[i, df.columns.get_loc('signal_command')] = 'hold'
                current_position == 0
            else:
                last_deal_price = price_now
                
            prev_command = df.iloc[i, df.columns.get_loc('signal_command')]


    signal_df = df[df['signal_command'] != 'hold']
    return df