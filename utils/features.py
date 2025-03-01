import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
import talib
import numpy as np
from talib import MA_Type

def add_features(source_df: pd.DataFrame) -> pd.DataFrame:
    # scaler = MinMaxScaler()
    for ticker in source_df.index.get_level_values(0).unique():
            # ticker_df = source_df[source_df['ticker'] == ticker]
            ticker_df = source_df.xs(ticker, level='ticker').copy() 
            # ticker_df['scaled_close_price'] = scaler.fit_transform(ticker_df[['close']])

            ticker_df['tema'] = talib.TEMA(ticker_df['close'], timeperiod=24)
            ticker_df['macd'], ticker_df['macd_signal_line'], ticker_df['macd_hist'] = talib.MACD(ticker_df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            ticker_df["bb_upper"], _ , ticker_df["bb_lower"] = talib.BBANDS(ticker_df['close'], matype=MA_Type.EMA)

          
            # Создаем сигналы для покупки и продажи
            # Добавляем SMA/EMA для фильтрации тренда
            ticker_df["SMA_50"] = talib.SMA(ticker_df["close"], timeperiod=50)
            ticker_df["SMA_20"] = talib.SMA(ticker_df["close"], timeperiod=20)
            ticker_df["SMA_10"] = talib.SMA(ticker_df["close"], timeperiod=10)
            ticker_df["SMA_200"] = talib.SMA(ticker_df["close"], timeperiod=200)
            ticker_df["EMA_50"] = talib.EMA(ticker_df["close"], timeperiod=50)
            ticker_df["SMA_30"] = talib.EMA(ticker_df["close"], timeperiod=30)
            ticker_df["SMA_60"] = talib.EMA(ticker_df["close"], timeperiod=60)
         

            ticker_df["EMA_9"] = talib.EMA(ticker_df["close"], timeperiod=9)
            ticker_df["EMA_21"] = talib.EMA(ticker_df["close"], timeperiod=21)    
            ticker_df["EMA_20"] = talib.EMA(ticker_df["close"], timeperiod=20)            
            ticker_df['RSI'] =  talib.RSI(ticker_df["close"], timeperiod=30)
            ticker_df['DX'] =  talib.DX(ticker_df["high"], ticker_df['low'], ticker_df['close'], timeperiod=30)


            ticker_df['OBV'] = talib.OBV(ticker_df['close'], ticker_df['volume'])
            ticker_df['AD'] = talib.AD(ticker_df['high'], ticker_df['low'], ticker_df['close'], ticker_df['volume'])
            ticker_df['MFI'] = talib.MFI(ticker_df['high'], ticker_df['low'], ticker_df['close'], ticker_df['volume'], timeperiod=14)

            ticker_df['VO'] = volume_oscillator(ticker_df['volume'])
            ticker_df['mean_close'] = ticker_df['close'].rolling(window=7, center=True).mean()


            ticker_df['Stochastic_K'], ticker_df['Stochastic_D'] = talib.STOCH(ticker_df['high'], ticker_df['low'], ticker_df['close'],
                                                          fastk_period=14, slowk_period=3, slowk_matype=0, 
                                                          slowd_period=3, slowd_matype=0)
            ticker_df['CCI'] = talib.CCI(ticker_df['high'], ticker_df['low'], ticker_df['close'], timeperiod=30)
            ticker_df['ATR'] = talib.ATR(ticker_df['high'], ticker_df['low'], ticker_df['close'], timeperiod=14)



            ticker_df['SMA_200_50'] = ticker_df['SMA_200'] - ticker_df['SMA_50']
            ticker_df['SMA_50_20'] = ticker_df['SMA_50'] - ticker_df['SMA_20']
            ticker_df['BB_UPPER_CLOSE'] = ticker_df['bb_upper'] - ticker_df['close']
            ticker_df['BB_LOWER_CLOSE'] = ticker_df['bb_lower'] - ticker_df['close']
            
            ticker_df['week_day'] = ticker_df.index.weekday
            ticker_df['year_day'] = ticker_df.index.day_of_year


            ticker_df.ffill(inplace=True)

            
            add_optimal_signals(ticker_df, price_column='close' , signal_column='train_signal_command', threshold=0.02)
            add_optimal_signals2(ticker_df, price_column='close',  signal_column='train_signal_command_2', threshold=0.02)
            ticker_df['train_signal_command_3'] = ticker_df['train_signal_command_2']
            fill_forward_until_next_nonzero(ticker_df, column='train_signal_command_3', offset=3)
            ticker_df['daily_return_volume'] = ticker_df['daily_return'] * ticker_df['volume']

            # Возвращаем данные в оригинальный датафрейм по всем ценным бумагам
            # source_df.loc[source_df['ticker'] == ticker, 'scaled_close_price'] = ticker_df['scaled_close_price']
            source_df.loc[(ticker, ticker_df.index), 'macd'] = ticker_df['macd'].values
            source_df.loc[(ticker, ticker_df.index), 'macd_signal_line'] = ticker_df['macd_signal_line'].values
            source_df.loc[(ticker, ticker_df.index), 'macd_hist'] = ticker_df['macd_hist'].values
            source_df.loc[(ticker, ticker_df.index), 'tema'] = ticker_df['tema'].values
            source_df.loc[(ticker, ticker_df.index), 'RSI'] = ticker_df['RSI'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_50'] = ticker_df['EMA_50'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_50'] = ticker_df['SMA_50'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_200'] = ticker_df['SMA_200'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_9'] = ticker_df['EMA_9'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_21'] = ticker_df['EMA_21'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_20'] = ticker_df['SMA_20'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_20'] = ticker_df['EMA_20'].values
            source_df.loc[(ticker, ticker_df.index), 'mean_close'] = ticker_df['mean_close'].values
            # source_df.loc[(ticker, ticker_df.index), 'signal_command'] = ticker_df['signal_command'].values
            source_df.loc[(ticker, ticker_df.index), 'bb_upper'] = ticker_df['bb_upper'].values
            source_df.loc[(ticker, ticker_df.index), 'bb_lower'] = ticker_df['bb_lower'].values
            
            source_df.loc[(ticker, ticker_df.index), 'OBV'] = ticker_df['OBV'].values
            source_df.loc[(ticker, ticker_df.index), 'AD'] = ticker_df['AD'].values
            source_df.loc[(ticker, ticker_df.index), 'MFI'] = ticker_df['MFI'].values
            source_df.loc[(ticker, ticker_df.index), 'VO'] = ticker_df['VO'].values
            source_df.loc[(ticker, ticker_df.index), 'Stochastic_K'] = ticker_df['Stochastic_K'].values
            source_df.loc[(ticker, ticker_df.index), 'CCI'] = ticker_df['CCI'].values
            source_df.loc[(ticker, ticker_df.index), 'Stochastic_D'] = ticker_df['Stochastic_D'].values
            source_df.loc[(ticker, ticker_df.index), 'train_signal_command'] = ticker_df['train_signal_command'].values
            source_df.loc[(ticker, ticker_df.index), 'train_signal_command_2'] = ticker_df['train_signal_command_2'].values
            source_df.loc[(ticker, ticker_df.index), 'train_signal_command_3'] = ticker_df['train_signal_command_3'].values
            source_df.loc[(ticker, ticker_df.index), 'daily_return'] = ticker_df['daily_return'].values
            source_df.loc[(ticker, ticker_df.index), 'ATR'] = ticker_df['ATR'].values
            source_df.loc[(ticker, ticker_df.index), 'DX'] = ticker_df['DX'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_30'] = ticker_df['SMA_30'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_10'] = ticker_df['SMA_10'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_60'] = ticker_df['SMA_60'].values


            source_df.loc[(ticker, ticker_df.index), 'SMA_200_50'] = ticker_df['SMA_200_50'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_50_20'] = ticker_df['SMA_50_20'].values
            source_df.loc[(ticker, ticker_df.index), 'BB_UPPER_CLOSE'] = ticker_df['BB_UPPER_CLOSE'].values
            source_df.loc[(ticker, ticker_df.index), 'BB_LOWER_CLOSE'] = ticker_df['BB_LOWER_CLOSE'].values
            source_df.loc[(ticker, ticker_df.index), 'daily_return_volume'] = ticker_df['daily_return_volume'].values
            source_df.loc[(ticker, ticker_df.index), 'week_day'] = ticker_df['week_day'].values
            source_df.loc[(ticker, ticker_df.index), 'year_day'] = ticker_df['year_day'].values



    # source_df['daily_return'] = source_df.groupby('ticker')['close'].pct_change()
    return source_df[1:]

def volume_oscillator(volume, short_period=5, long_period=14):
    short_ma = talib.SMA(volume, timeperiod=short_period)
    long_ma = talib.SMA(volume, timeperiod=long_period)
    return ((short_ma - long_ma) / long_ma) * 100


def add_optimal_signals2(ticker_df, price_column='mean_close' , signal_column='train_signal_command_2', threshold=0.005):
    ticker_df['daily_return'] = ticker_df[price_column].pct_change()
    ticker_df[signal_column] = 0
    ticker_df.loc[ticker_df['daily_return'] > threshold, signal_column] = 1
    ticker_df.loc[ticker_df['daily_return'] < -threshold, signal_column] = -1
    ticker_df[signal_column] = ticker_df[signal_column].shift(-1)
    prev_signal = 0
    for i, row in ticker_df.loc[ticker_df[signal_column] !=0].iterrows():
        signal_command = row[signal_column]
        if signal_command == prev_signal:
            ticker_df.loc[i, signal_column] = 0
        prev_signal = signal_command
    # ticker_df.loc[ticker_df[signal_column].duplicated(keep='first'), signal_column] = 0



def add_optimal_signals(df, price_column = 'mean_close', signal_column='train_signal_command' , threshold=0.001):
    df['daily_return'] = df[price_column].pct_change()
    df[signal_column]  = 0
    df.loc[df['daily_return'] > threshold, signal_column] = 1
    df.loc[df['daily_return'] < -threshold, signal_column] = -1
    df[signal_column] = df[signal_column].shift(-1)



def fill_forward_until_next_nonzero(df, column, offset=3):
    filled_values = df[column].copy()
    last_nonzero_index = None
    # last_signal = 0

    for i in range(len(df)):
        if df[column].iloc[i] != 0:
            # last_signal = df[column].iloc[i]
            if last_nonzero_index is not None:
                fill_start = last_nonzero_index + 1
                fill_end = max(i - offset, fill_start)
                filled_values.iloc[fill_start:fill_end] = df[column].iloc[last_nonzero_index]
            last_nonzero_index = i

    if last_nonzero_index is not None:
        fill_start = last_nonzero_index + 1
        fill_end = len(df)
        filled_values.iloc[fill_start:fill_end] = df[column].iloc[last_nonzero_index]

    return filled_values

# Apply the function to fill forward non-zero values until 3 rows before the next non-zero value
