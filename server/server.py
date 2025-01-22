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
                self.counter += 1
            else:
                self.counter = 0

        self.val_losses.append(val_loss)
        self.losses.append(loss)

        if self.counter >= self.patience:
            self.model.stop_training = True

class PredictionRequest(BaseModel):
    stock: str
    dateRange: dict
    forecastDays: conint(ge=1)
    selectedMacro: list

def load_macro_data(selected_macro, start_date, end_date):
    macro_data = {}
    for macro in selected_macro:
        try:
            data = yf.download(macro, start=start_date, end=end_date, progress=False)['Close']
            macro_data[macro] = data
        except Exception as e:
            logging.error(f"Ошибка при загрузке макроэкономического показателя {macro}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при загрузке данных для {macro}")
    macro_df = pd.DataFrame(macro_data)
    macro_df.dropna(inplace=True)
    return macro_df

def create_dataset(data, look_back):
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
        
        print(ticker, start_date, end_date, forecast_days, selected_macro)

        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
        print("Stock data:", stock_data.head())

        macro_data = load_macro_data(selected_macro, start_date, end_date)
        print("Macro data:", macro_data.head())
        
        df = pd.concat([stock_data, macro_data], axis=1).dropna()

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

        model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

        last_window = data_scaled[-look_back:]
        predictions = []

        for _ in range(forecast_days):
            current_input = last_window.reshape(1, look_back, x_train.shape[2])
            prediction = model.predict(current_input, verbose=0)
            predictions.append(prediction[0][0])

            new_row = last_window[-1:].copy()
            new_row[0, 0] = prediction[0][0]
            last_window = np.vstack((last_window[1:], new_row))

        predictions = np.array(predictions).reshape(-1, 1)
        predictions_original_scale = scaler.inverse_transform(
            np.hstack([predictions, np.zeros((len(predictions), x_train.shape[2] - 1))])
        )[:, 0]

        forecast_dates = pd.date_range(start=end_date, periods=forecast_days, freq='B')

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df.iloc[:, 0], label='Фактические значения', color='blue')
        plt.plot(forecast_dates, predictions_original_scale, label='Предсказанные значения', color='orange')
        plt.title(f'Фактические и предсказанные значения для {ticker}')
        plt.xlabel('Дата')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type='image/png')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
