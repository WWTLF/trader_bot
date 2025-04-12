from utils.download import get_last_month
from utils.features import add_features
from utils.signals import add_signals
from datetime import datetime
from dateutil.relativedelta import relativedelta
from forecast.lstm_predictor import LSTMPredictorService
from forecast.tsmixer_predictor import TSMixerPredictorService
from forecast.lstm_verifier import LSTMVerifierService
from sklearn.preprocessing import MinMaxScaler
from repositories.extra_feature import ExtraFeatureRepository
from models.extra_features import ExtraFeature
from services.decistion_service import DecisionService
from pykalman import KalmanFilter
import psycopg



def decide(current_day: datetime, conn: psycopg.Connection):



    # Все данные и гипперпараметры моделей хранятся в PostgreSQL. Установим соединения с БД.
    # conn = get_conn()
    # stocks = ['AAPL', 'GOOG','AMZN', 'MSFT', 'AMD', 'NVDA', 'TSLA']
    stocks = ['GOOG']
    extra_feature_repo = ExtraFeatureRepository(conn)

    df = get_last_month(conn, current_day, 6)
    df = add_features(df)
    
    for stock_name in stocks:
        skip = False
        print("Processing", stock_name, current_day)
        signals_df = df.loc[stock_name]
        ds = DecisionService(stock_name, conn=conn)
        ds.deserialize()

        # Данный метод добавляет разные сигналы на основе технических индекаторов и пару "идеальных" сигналов на основе истории цены для обучения моделей. 
        add_signals(signals_df)
        signals_df['pct_real_close'] = signals_df['close'].pct_change(1)


        lstm_predictor = LSTMPredictorService(stock_name, conn)
        lstm_predictor.deserizalize(signals_df, 0.0)
        lstm_model_config = lstm_predictor.get_mdoel_config()

        lstm_offset = lstm_model_config.config['offset']

        lstm_scalerY = MinMaxScaler()
        lstm_scalerY.min_ = lstm_model_config.config['scalerMinY']
        lstm_scalerY.scale_ =  lstm_model_config.config['scalerSlaceY']


        tsmixer_predictor = TSMixerPredictorService(stock_name, conn)
        tsmixer_predictor.deserizalize(signals_df, 0.0)
        ts_model_config = tsmixer_predictor.get_mdoel_config()

        ts_offset = ts_model_config.config['offset']
        lstm_verifier = LSTMVerifierService(stock_name, conn)
        lstm_verifier.deserizalize(signals_df, 0.0)
        ts_scalerY = MinMaxScaler()
        ts_scalerY.min_ = ts_model_config.config['scalerMinY']
        ts_scalerY.scale_ =  ts_model_config.config['scalerSlaceY']
        # signals_df['scaled_close'] = lstm_scalerY.transform(signals_df[['close']])
        # Сначала предсказываем цену на T+3
        lstm_pred_t3_close, _ = lstm_predictor.predict(current_day)
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day + relativedelta(days=lstm_offset), 'lstm_pred_t3_close', lstm_pred_t3_close))

        ts_mixer_t3_close, _ = tsmixer_predictor.predict(current_day)
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day + relativedelta(days=ts_offset), 'ts_mixer_t3_close', ts_mixer_t3_close))

        lstm_pred_t3_close_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day + relativedelta(days=3), 'lstm_pred_t3_close')
        if lstm_pred_t3_close_feature is not None:
           lstm_pred_t3_close =  lstm_pred_t3_close_feature.feature_value
        else:
            skip = True
            lstm_pred_t3_close = 0.0

        ts_mixer_t3_close_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day + relativedelta(days=3), 'ts_mixer_t3_close')
        if ts_mixer_t3_close_feature is not None:
            ts_mixer_t3_close = ts_mixer_t3_close_feature.feature_value
        else:
            skip = True
            ts_mixer_t3_close = 0.0

        mid_pred_t3_close = 0.5 * lstm_pred_t3_close + 0.5 * ts_mixer_t3_close
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day + relativedelta(days=3), 'mid_pred_t3_close', mid_pred_t3_close))


        filtered_mid_price_t0_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day, 'filtered_mid_price_t3')
        if filtered_mid_price_t0_feature is not None:
            filtered_mid_price_t0 = filtered_mid_price_t0_feature.feature_value
        else:
            skip = True
            filtered_mid_price_t0 = 0.0

        filtered_mid_price_t2_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day + relativedelta(days=2), 'filtered_mid_price_t3')
        if filtered_mid_price_t2_feature is not None:
            filtered_mid_price_t2 = filtered_mid_price_t2_feature.feature_value
        else:
            skip = True
            filtered_mid_price_t2 = 0.0

        filtered_covariance_t2_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day + relativedelta(days=2), 'filtered_covariance_t3')
        if filtered_covariance_t2_feature is not None:
            filtered_covariance_t2 = filtered_covariance_t2_feature.feature_value
        else:
            skip = True
            filtered_covariance_t2 = 0.0


        kf = KalmanFilter(
            transition_matrices=[1],  # Цена меняется плавно
            observation_matrices=[1],  # Мы наблюдаем цену напрямую
            initial_state_mean=filtered_mid_price_t2,  # Первая цена (например, средняя)
            initial_state_covariance=0.01,  # Начальная неопределенность (уменьшена)
            observation_covariance=0.01,  # Ошибка предсказания LSTM (чем меньше, тем сильнее сглаживание)
            transition_covariance=0.0005  # Насколько плавно менять (чем меньше, тем более плавный тренд)
            )

        # Сглаживаем среднюю цену
        filtered_mid_price_t3, filtered_covariance_t2 = kf.filter_update(
            [[filtered_mid_price_t2]],  # Последнее отфильтрованное значение
            [[filtered_covariance_t2]],  # Ковариация
            observation= [[mid_pred_t3_close]]
        )
        filtered_mid_price_t3 = filtered_mid_price_t3[0,0]
        filtered_covariance_t2 = filtered_covariance_t2[0,0]

        if not skip:
            pct_pred_close = (filtered_mid_price_t3 - filtered_mid_price_t2) / filtered_mid_price_t2
        else:
            pct_pred_close = filtered_mid_price_t3

        # print(filtered_mid_price)
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day+relativedelta(days=3), 'pct_pred_close', pct_pred_close))
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day+relativedelta(days=3), 'filtered_mid_price_t3', filtered_mid_price_t3))
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day+relativedelta(days=3), 'filtered_covariance_t3', filtered_covariance_t2))

        new_signal = 0
        # Простое правило, выдающее предварительный сигнал
        if filtered_mid_price_t3 > filtered_mid_price_t0:
            new_signal = 1
        elif filtered_mid_price_t3 < filtered_mid_price_t0:
            new_signal = -1
    
        
        # prev_signal_feature = extra_feature_repo.get_one_by_ticker_and_date(stock_name, current_day + relativedelta(days=-1), 'signal')
        # if prev_signal_feature is not None:
        #     new_signal = prev_signal_feature.feature_value
        #     print("New signal: ", new_signal)
        #     # result_df.iloc[i, result_df.columns.get_loc('signal')] = new_signal
        #     # prev_signal = new_signal
        # else:
        #     skip = True
        #     new_signal = 0
        
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day, 'signal', new_signal))


        # Решаем доверять ли предсказанию на данный день (Если мы не доверяем, то не доверяем все три дня)
        if not skip:
            not_trust = lstm_verifier.predict(current_day)
        else:
            not_trust = 1.0
        extra_feature_repo.save(ExtraFeature(None, stock_name, current_day, 'not_trust', not_trust))

        # not_trust = 0.0 # Для предварительно обучения всегда 0, затем нужно закомментировать
        if skip == True:
            not_trust = 1.0
            new_signal = 0.0

        price_day = current_day.strftime("%Y-%m-%d")
        if (price_day in signals_df.index) and (not skip):
            price = signals_df.loc[price_day]['close']
            is_close, final_signal = ds.decide(current_day, new_signal, price, 1, not_trust)
            # result_df.iloc[i, result_df.columns.get_loc('is_close')] = is_close
            extra_feature_repo.save(ExtraFeature(None, stock_name, current_day, 'is_close', float(is_close)))
            # result_df.iloc[i, result_df.columns.get_loc('final_signal')] = final_signal
            extra_feature_repo.save(ExtraFeature(None, stock_name, current_day, 'final_signal', final_signal))
            
        else:
            print("Not enough data for ", price_day)

        # print(signals_df.info())


