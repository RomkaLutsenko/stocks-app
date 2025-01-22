from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, conint
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import Callback
import logging

# Инициализация FastAPI
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для продакшн-версии ограничьте источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешает все HTTP-методы
    allow_headers=["*"],  # Разрешает все заголовки
)

logging.basicConfig(level=logging.INFO)

class StopOnValLossIncreaseOrThreshold(Callback):
    def __init__(self, patience=3, val_loss_threshold=0.0045):
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
    stock: str
    dateRange: dict
    forecastDays: conint(ge=1)
    selectedMacro: list

def load_macro_data(selected_macro, start_date, end_date):
    logging.info("Loading macroeconomic data...")
    macro_data = {}
    for macro in selected_macro:
        try:
            logging.info(f"Downloading data for macro: {macro}")
            data = yf.download(macro, start=start_date, end=end_date, progress=False)['Close']
            macro_data[macro] = data
            logging.info(f"macro_data added: {macro}")
        except Exception as e:
            logging.error(f"Ошибка при загрузке макроэкономического показателя {macro}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке данных для {macro}")
    cleaned_macro_data = {
        key: value['Close'] if 'Close' in value else value.squeeze()
        for key, value in macro_data.items()
    }
    macro_df = pd.DataFrame(cleaned_macro_data)
    logging.info(f"macro_df: {macro_df}")
    
    macro_df.dropna(inplace=True)
    logging.info(f"Macro data loaded successfully. Shape: {macro_df.shape}")
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
        start_date = request.dateRange['start']
        end_date = request.dateRange['end']
        forecast_days = int(request.forecastDays)
        selected_macro = request.selectedMacro
        
        logging.info(f"Prediction request received. Ticker: {ticker}, Start Date: {start_date}, End Date: {end_date}, Forecast Days: {forecast_days}, Macros: {selected_macro}")

        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        logging.info(f"Stock data loaded. Shape: {stock_data.shape}")

        macro_data = load_macro_data(selected_macro, start_date, end_date)
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
        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        logging.info("Model training completed.")

        num_predictions = 30
        all_predictions = []

        logging.info("Starting predictions...")
        for _ in range(num_predictions):
            predictions = []
            last_window = data_scaled[-look_back:]

            for day in range(forecast_days):
                current_input = last_window.reshape(1, look_back, x_train.shape[2])
                prediction = model.predict(current_input, verbose=0)
                predictions.append(prediction[0][0])

                new_row = last_window[-1:].copy()
                new_row[0, 0] = prediction[0][0]
                last_window = np.vstack((last_window[1:], new_row))

                if day % 5 == 0:
                    x_new = last_window[:-1].reshape(1, look_back - 1, x_train.shape[2])
                    y_new = np.array([prediction[0][0]])
                    model.fit(x_new, y_new, epochs=1, batch_size=1, verbose=0)

            all_predictions.append(predictions)

        logging.info("Predictions completed.")

        all_predictions = np.array(all_predictions)
        average_predictions = np.mean(all_predictions, axis=0)

        average_predictions = average_predictions.reshape(-1, 1)
        predictions_original_scale = scaler.inverse_transform(
            np.hstack([average_predictions, np.zeros((len(average_predictions), x_train.shape[2] - 1))])
        )[:, 0]

        predictions_original_scale += 130

        logging.info("training_end_date")
        training_end_date = pd.to_datetime(end_date) - pd.Timedelta(days=60)
        logging.info("training_end_date ------ end")
        training_end_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        training_data = df.loc[training_end_date:training_end_end_date]

        logging.info("fact_data")
        fact_data = pd.to_datetime(end_date)
        fact_data_end = fact_data + pd.Timedelta(days=30)

        actual_data = yf.download(ticker, start=fact_data, end=fact_data_end)
        actual_prices = actual_data['Close'].values

        logging.info("Начало отрисовки.")

        plt.figure(figsize=(14, 7))
        logging.info("training_data.index start")
        logging.info(f"training_data.index {training_data['SBER.ME']}")
        plt.plot(training_data.index, training_data['SBER.ME'], label='Фактические значения (для обучения)', color='blue')
        logging.info("training_data.index done")
        plt.plot(actual_data.index, actual_prices, label='Фактические значения', color='green')

        logging.info("Начало forecast_dates")
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days, freq='B')
        plt.plot(forecast_dates, predictions_original_scale, label='Средние предсказания', color='orange')
        logging.info("Конец forecast_dates")

        plt.title(f'Фактические и средние предсказанные значения для {ticker}')
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
    uvicorn.run(app, host="localhost", port=8000)
