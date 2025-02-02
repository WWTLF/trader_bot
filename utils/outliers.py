import pandas as pd

def get_rid_of_outliers(source_df: pd.DataFrame): 
    for ticker in source_df['ticker'].unique():
        ticker_df = source_df[source_df['ticker'] == ticker]
        # Вычисление межквартильного размаха (IQR)
        Q1 = ticker_df['close'].quantile(0.25)  # Первый квартиль (25-й процентиль)
        Q3 = ticker_df['close'].quantile(0.75)  # Третий квартиль (75-й процентиль)
        IQR = Q3 - Q1                        # Межквартильный размах

        # Определение границ для выбросов
        lower_bound = Q1 - 1.5 * IQR  # Нижняя граница
        upper_bound = Q3 + 1.5 * IQR  # Верхняя граница

        # Пометка выбросов
        ticker_df['is_outlier'] = (ticker_df['close'] < lower_bound) | (ticker_df['close'] > upper_bound)
        window = 5  # Количество дней для среднего
        ticker_df['original_price'] = ticker_df['close']
        ticker_df.loc[ticker_df['is_outlier'], 'close'] =  ticker_df['original_price'].rolling(window=window, center=True).mean()
        ticker_df.drop(columns=['original_price'], inplace=True)
        ticker_df.ffill(inplace=True)
        source_df.loc[source_df['ticker'] == ticker, 'close'] = ticker_df['close']
        source_df.loc[source_df['ticker'] == ticker, 'is_outlier'] = ticker_df['is_outlier']