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
            ticker_df["SMA_200"] = talib.SMA(ticker_df["close"], timeperiod=200)
            ticker_df["EMA_50"] = talib.EMA(ticker_df["close"], timeperiod=50)
         

            ticker_df["EMA_9"] = talib.EMA(ticker_df["close"], timeperiod=9)
            ticker_df["EMA_21"] = talib.EMA(ticker_df["close"], timeperiod=21)        
            ticker_df['RSI'] =  talib.RSI(ticker_df["close"], timeperiod=14)

            ticker_df['mean_close'] = ticker_df['close'].rolling(window=7, center=True).mean()
            ticker_df.ffill(inplace=True)


            
            ticker_df['signal_command'] = None
            add_optimal_signals(ticker_df, price_column='mean_close', threshold=0.01)

            # Возвращаем данные в оригинальный датафрейм по всем ценным бумагам
            # source_df.loc[source_df['ticker'] == ticker, 'scaled_close_price'] = ticker_df['scaled_close_price']
            source_df.loc[(ticker, ticker_df.index), 'macd'] = ticker_df['macd'].values
            source_df.loc[(ticker, ticker_df.index), 'macd_signal_line'] = ticker_df['macd_signal_line'].values
            source_df.loc[(ticker, ticker_df.index), 'tema'] = ticker_df['tema'].values
            source_df.loc[(ticker, ticker_df.index), 'RSI'] = ticker_df['RSI'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_50'] = ticker_df['EMA_50'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_50'] = ticker_df['SMA_50'].values
            source_df.loc[(ticker, ticker_df.index), 'SMA_200'] = ticker_df['SMA_200'].values
            source_df.loc[(ticker, ticker_df.index), 'signal_command'] = ticker_df['signal_command'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_9'] = ticker_df['EMA_9'].values
            source_df.loc[(ticker, ticker_df.index), 'EMA_21'] = ticker_df['EMA_21'].values
            source_df.loc[(ticker, ticker_df.index), 'mean_close'] = ticker_df['mean_close'].values
            # source_df.loc[(ticker, ticker_df.index), 'signal_command'] = ticker_df['signal_command'].values
            source_df.loc[(ticker, ticker_df.index), 'bb_upper'] = ticker_df['bb_upper'].values
            source_df.loc[(ticker, ticker_df.index), 'bb_lower'] = ticker_df['bb_lower'].values




    source_df['daily_return'] = source_df.groupby('ticker')['close'].pct_change()
    return source_df[1:]


def add_optimal_signals(df, price_column = 'mean_close', threshold=0.05):
    df['signal_command'] = 'hold'
    prev_trend = 0
    prev_price = df.iloc[2]['mean_close']
    # if prev_price == 0.0:
    #     prev_price = 0.0001
    for index, row in df[3:].iterrows():
        current_price = row['mean_close']
        pct = (current_price - prev_price) / prev_price
        if pct > 0 and prev_trend !=1:
            prev_trend = 1
            df.loc[index, 'signal_command'] = "draft"
        elif pct < 0 and prev_trend != -1:
            prev_trend = -1
            df.loc[index, 'signal_command'] = "draft"    
        prev_price = current_price

    prev_price = df.iloc[2]['mean_close']
    # print(prev_price)
    draft_signals = df[df['signal_command'] == "draft"]
    for index, row in draft_signals.iterrows():
        current_price = row['mean_close']
        # print("current price ", current_pruce)
        pct = abs(current_price - prev_price) / prev_price
        # print("pct           ", pct)
        if pct > threshold: 
            df.loc[index, 'signal_command'] = 'draft'
        else:
            df.loc[index, 'signal_command'] = 'hold'
        prev_price = current_price


    draft_signals = df[df['signal_command'] == "draft"]
    if len(draft_signals) < 2:
        return
    # print(draft_signals.iloc[0]['mean_close'] ,draft_signals.iloc[1]['mean_close'] )
    if draft_signals.iloc[0]['mean_close'] > draft_signals.iloc[1]['mean_close']:
        df.loc[draft_signals.index[0], 'signal_command'] = 'sell'
    else:
        df.loc[draft_signals.index[0],'signal_command'] = 'buy'

    prev_deal_signal = df.loc[draft_signals.index[0], 'signal_command'] 
    prev_deal_price = df.loc[draft_signals.index[0], 'mean_close'] 

    loc_index = 0
    draft_signals = df[df['signal_command'] == "draft"]
    for index, row in draft_signals[:-1].iterrows():
        current_price = row['mean_close']
        next_price = draft_signals.iloc[loc_index + 1]['mean_close']
        # print(loc_index, prev_deal_price, current_price , next_price)
        if current_price > prev_deal_price:
            if prev_deal_signal == "buy" and next_price < current_price:
                df.loc[index, 'signal_command'] = 'sell'
                prev_deal_signal = 'sell'
            else:
                df.loc[index, 'signal_command'] = 'hold'
        else:
            if prev_deal_signal == "sell" and next_price > current_price:
                df.loc[index, 'signal_command'] = 'buy'
                prev_deal_signal = 'buy'
            else:
                df.loc[index, 'signal_command'] = 'hold'
        prev_deal_price = current_price
        loc_index = loc_index + 1

    draft_signals = df[df['signal_command'] == "draft"]
    df.loc[draft_signals.index, "signal_command"] = "hold"