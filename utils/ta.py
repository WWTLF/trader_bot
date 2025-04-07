
import pandas as pd
import numpy as np

def TEMA(series, period):
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3

def MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - signal_line
    return macd, signal_line, macd_hist

def BBANDS(series, window=20, num_std=2):
    rolling_mean = series.ewm(span=window, adjust=False).mean()
    rolling_std = series.ewm(span=window, adjust=False).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return upper, rolling_mean, lower

def SMA(series, period):
    return series.rolling(window=period).mean()

def EMA(series, period):
    return series.ewm(span=period, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def DX(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx

def OBV(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()

def AD(high, low, close, volume):
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.replace([np.inf, -np.inf], 0).fillna(0)
    return (clv * volume).cumsum()

def MFI(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    pos_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    pos_sum = pos_flow.rolling(window=period).sum()
    neg_sum = neg_flow.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi

def ATR(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def CCI(high, low, close, period=20):
    typical_price = (high + low + close) / 3
    mean_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (typical_price - mean_tp) / (0.015 * mean_deviation)

def STOCH(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def volume_oscillator(volume, short_period=5, long_period=14):
    short_ma = volume.rolling(window=short_period).mean()
    long_ma = volume.rolling(window=long_period).mean()
    return ((short_ma - long_ma) / long_ma) * 100
