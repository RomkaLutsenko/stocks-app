# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
from datetime import datetime, timedelta
from tinkoff.invest import Client, CandleInterval

import logging
from typing import ClassVar

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

epochs = 80
API_KEY  = "t.BtkoR-J4W1Zv_ZLwNySaplKK3BDC3NnDQk0RGbnx9U57uiREVKsVDYQbq7lasJzEFN4EwUtO7c_FtrJkCRhCag"
max_period = timedelta(days=365)



class StopOnValLossIncreaseOrThreshold(Callback):
    def __init__(self, patience=3, val_loss_threshold=0.002):
        super(StopOnValLossIncreaseOrThreshold, self).__init__()
        self.patience = patience
        self.val_loss_threshold = val_loss_threshold
        self.val_losses = []
        self.losses = []
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if val_loss < self.val_loss_threshold:
            self.model.stop_training = True
            return

        if len(self.val_losses) > 0:
            if val_loss > self.val_losses[-1] and loss < self.losses[-1]:
                logging.info(f"Validation loss below threshold: {val_loss}")
                self.counter += 1
            else:
                self.counter = 0

        self.val_losses.append(val_loss)
        self.losses.append(loss)

        if self.counter >= self.patience:
            logging.info(f"Stopping training after {self.counter} epochs with no improvement.")
            self.model.stop_training = True

class PredictionRequest(BaseModel):
    stock: dict
    dateRange: dict
    selectedMacro: list
    daysRange: str
    

def daterange(start_dt, end_dt, delta):
    """Генератор диапазонов дат для разбиения большого периода на части."""
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + delta, end_dt)
        yield current_start, current_end
        current_start = current_end

def load_macro_data(selected_macro, learning_start_date, learning_end_date):
    logging.info("Loading macroeconomic data...")
    macro_data = {}
    
    with Client(API_KEY) as client:
        for macro in selected_macro:
            name = macro["name"]
            symbol = macro["symbol"]
            if symbol.endswith(".xlsx"):
                # Чтение данных из Excel-файла (панические индексы)
                try:
                    panic_data = pd.read_excel(symbol)
                    panic_data.reset_index(inplace=True)
                    panic_data.set_index("Date", inplace=True)
                    panic_data.drop(columns=["index"], inplace=True, errors='ignore')
                    series = panic_data.squeeze()
                    series.name = name
                    macro_data[name] = series
                except Exception as e:
                    print(f"Ошибка при чтении файла {symbol} для {name}: {e}")
            else:
                # Сбор данных через Tinkoff Invest API
                all_candles = []
                for period_start, period_end in daterange(learning_start_date, learning_end_date, max_period):
                    try:
                        response = client.market_data.get_candles(
                            figi=symbol,
                            from_=period_start,
                            to=period_end,
                            interval=CandleInterval.CANDLE_INTERVAL_DAY
                        )
                        if response.candles:
                            all_candles.extend(response.candles)
                    except Exception as e:
                        print(f"Ошибка при запросе {name} с {period_start} по {period_end}: {e}")
                # Формирование Series: индекс — дата, значение — цена закрытия
                # Если API возвращает тип time с часовым поясом, извлекаем дату
                data = {}
                for candle in all_candles:
                    # Получаем дату из candle.time (например, методом .date())
                    date_key = candle.time.date()
                    # Считаем цену закрытия как сумму единиц и наносекунд (если требуется)
                    price = candle.close.units + candle.close.nano / 1_000_000_000
                    data[date_key] = price
                if data:
                    series = pd.Series(data, name=name)
                    macro_data[name] = series
                else:
                    print(f"Нет данных для {name} за указанный период.")
    
    # Преобразуем все Series в macro_data к datetime64 перед объединением
    for key in macro_data:
        macro_data[key].index = pd.to_datetime(macro_data[key].index)
    # Объединяем данные в DataFrame
    macro_df = pd.DataFrame(macro_data)

    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df.index.name = "Date"

    return macro_df

def create_dataset(data, look_back):
    logging.info("Creating dataset for model training...")
    x, y = [], []
    for i in range(look_back, len(data)):
        x.append(data[i - look_back:i, :])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        ticker = request.stock
        forecast_start_date = datetime.strptime(request.dateRange["start"], "%Y-%m-%d")
        forecast_end_date = datetime.strptime(request.dateRange["end"], "%Y-%m-%d")
        learning_start_date = datetime.strptime('2016-01-04', "%Y-%m-%d")
        learning_end_date = forecast_start_date + timedelta(days=1)
        forecast_days = (forecast_end_date - forecast_start_date).days
        selected_macro = request.selectedMacro
        left_right_range = int(request.daysRange)
        
        logging.info(f"Prediction request received. Ticker: {ticker}, Start Date: {request.dateRange['start']}, End Date: {request.dateRange['end']}, Forecast Days: {forecast_days}, Macros: {selected_macro}")

        all_candles = []
        with Client(API_KEY) as client:
            for period_start, period_end in daterange(learning_start_date, learning_end_date, max_period):
                try:
                    response = client.market_data.get_candles(
                        figi=ticker['symbol'],
                        from_=period_start,
                        to=period_end,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY
                    )
                    if response.candles:
                        all_candles.extend(response.candles)
                    print(f"Получены свечи с {period_start} по {period_end}")
                except Exception as e:
                    print(f"Ошибка при запросе свечей с {period_start} по {period_end}: {e}")

        # Вывод в нужном формате
        data = {
            candle.time.date(): candle.close.units + candle.close.nano / 1_000_000_000
            for candle in all_candles
        }
        stock_data = pd.DataFrame.from_dict(data, orient="index", columns=[ticker["name"]])
        stock_data.index.name = "Date"
        logging.info(f"Stock data loaded. Shape: {stock_data.shape}")

        macro_data = load_macro_data(selected_macro, learning_start_date, learning_end_date)
        # Определяем столбцы, пропуски в которых не надо заменять
        exclude_columns = {"Индекс паники отрицательный", "Индекс паники положительный", "Индекс паники по модулю"}
        # Заполняем NaN средними значениями только для нужных столбцов
        for col in macro_data.columns:
            if col not in exclude_columns:
                macro_data[col].fillna(macro_data[col].mean(), inplace=True)

        macro_data = macro_data.dropna(how='any')
        
        logging.info(f"Macro data loaded. Shape: {macro_data.shape}")
        
        df = pd.concat([stock_data, macro_data], axis=1).dropna()
        logging.info(f"Combined dataset created. Shape: {df.shape}")

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df.values)
        look_back = 60

        x_train, y_train = create_dataset(data_scaled[:int(len(data_scaled) * 0.7)], look_back)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = StopOnValLossIncreaseOrThreshold(patience=3)

        logging.info("Starting model training...")
        model.fit(x_train, y_train, epochs=50, batch_size=8, validation_split=0.9, callbacks=[early_stopping], verbose=0)
        logging.info("Model training completed.")

        predictions = []
        last_window = data_scaled[-look_back:]

        logging.info("Starting predictions...")
        for _ in range(forecast_days):
            current_input = last_window.reshape(1, look_back, x_train.shape[2])
            prediction = model.predict(current_input, verbose=0)
            predictions.append(prediction[0][0])
            
            new_row = last_window[-1:].copy()
            new_row[0, 0] = prediction[0][0]
            last_window = np.vstack((last_window[1:], new_row))

        logging.info("Predictions completed.")

        predictions = np.array(predictions).reshape(-1, 1)
        predictions_original_scale = scaler.inverse_transform(
            np.hstack([predictions, np.zeros((len(predictions), x_train.shape[2] - 1))])
        )[:, 0]

        logging.info("predictions_original_scale")

        forecast_end_extended = forecast_end_date + timedelta(days=left_right_range)
        logging.info(f"forecast_end_date {forecast_end_date}")
        logging.info(f"left_right_range {left_right_range}")
        all_actual_candles = []

        with Client(API_KEY) as client:
            for period_start, period_end in daterange(forecast_start_date, forecast_end_extended, max_period):
                try:
                    response = client.market_data.get_candles(
                        figi=ticker['symbol'],
                        from_=period_start,
                        to=period_end,
                        interval=CandleInterval.CANDLE_INTERVAL_DAY
                    )
                    if response.candles:
                        all_actual_candles.extend(response.candles)
                    print(f"Получены свечи с {period_start} по {period_end}")
                except Exception as e:
                    print(f"Ошибка при запросе свечей с {period_start} по {period_end}: {e}")
                    logging.error(f"Ошибка при запросе свечей с {period_start} по {period_end}: {e}")

        actual_data = {
            candle.time.date(): candle.close.units + candle.close.nano / 1_000_000_000
            for candle in all_actual_candles
        }
        logging.info(f"actual_data {actual_data}")

        actual_df = pd.DataFrame.from_dict(actual_data, orient="index", columns=[ticker["name"]])
        actual_df.index.name = "Date"

        ticker_name = ticker['name']
        actual_prices = actual_df[ticker_name].values

        logging.info(f"actual_prices: {actual_prices}")
        logging.info(f"left_right_range: {left_right_range}")

        logging.info(f"После загрузки actual_df, до начала отрисовки.")

        logging.info(f"Начало отрисовки.")
        plt.figure(figsize=(14, 7))

        logging.info(f"training_data.index {df[ticker_name]}")
        plt.plot(df.index[-left_right_range:], df[ticker_name].iloc[-left_right_range:], label='Фактические значения (для обучения)', color='blue')

        logging.info("training_data.index done")
        plt.plot(actual_df.index, actual_prices, label='Фактические значения', color='green')

        logging.info("Начало forecast_dates")
        forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_days, freq='B')
        plt.plot(forecast_dates, predictions_original_scale, label='Предсказанные значения', color='orange')
        logging.info("Конец forecast_dates")

        plt.title(f'Фактические и средние предсказанные значения для тикера {ticker_name}')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)
        logging.info("Нарисовалось.")

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        logging.info("Returning response...")
        return StreamingResponse(buf, media_type='image/png')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
