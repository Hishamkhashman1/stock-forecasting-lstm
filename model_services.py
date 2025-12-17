from typing import Tuple

import numpy as np
import pandas as pd
from keras import Input, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def build_lstm_model(
    input_shape: Tuple[int, int],
    units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    l2_lambda: float = 0.01,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units=units, return_sequences=True, kernel_regularizer=l2(l2_lambda)),
            Dropout(dropout_rate),
            LSTM(
                units=units // 2, return_sequences=False, kernel_regularizer=l2(l2_lambda)
            ),
            Dropout(dropout_rate),
            Dense(64, activation="relu", kernel_regularizer=l2(l2_lambda)),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_absolute_error", metrics=["mae"])
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: StandardScaler,
    *,
    batch_size: int = 50,
    epochs: int = 50,
    units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    l2_lambda: float = 0.01,
):
    print(
        "Training model with "
        f"units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, "
        f"batch_size={batch_size}, epochs={epochs}, l2_lambda={l2_lambda}"
    )
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
    )
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
        verbose=1,
    )

    predictions = model.predict(X_test)
    predictions_inv = target_scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    print("Original values (first 5):", y_test_inv[:5].flatten())
    print("Predicted values (first 5):", predictions_inv[:5].flatten())

    mae = mean_absolute_error(y_test_inv, predictions_inv)
    mse = mean_squared_error(y_test_inv, predictions_inv)

    return model, mae, mse


def forecast_trend(
    model: Sequential,
    stock_data,
    exchange_data,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    *,
    sequence_length: int = 120,
    forecast_days: int = 30,
):
    features = [
        "Adj Close",
        "SMA",
        "EMA",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Upper_BB",
        "Lower_BB",
        "ATR",
        "CCI",
        "Momentum",
        "Parabolic_SAR",
        "Volume",
    ]
    stock_data = stock_data[features]
    exchange_data = exchange_data[["Adj Close"]].rename(columns={"Adj Close": "Exchange_Adj_Close"})

    combined_df = pd.concat([stock_data, exchange_data], axis=1)
    column_count = combined_df.shape[1]

    last_sequence = combined_df.values[-sequence_length:]
    current_sequence = last_sequence.copy()
    forecast = []

    print(f"Last sequence for forecasting: {last_sequence}")

    for _ in range(forecast_days):
        scaled_sequence = feature_scaler.transform(current_sequence)
        model_input = scaled_sequence.reshape(1, sequence_length, column_count)
        prediction = model.predict(model_input, verbose=0)
        forecast.append(prediction[0, 0])

        new_row = current_sequence[-1].copy()
        new_row[0] = prediction[0, 0]
        current_sequence = np.vstack([current_sequence[1:], new_row])

    forecast_inv = target_scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    print(f"Forecasted values: {forecast_inv.flatten()}")
    return forecast_inv


def determine_trend_and_recommendation(forecasted_prices):
    initial_price = forecasted_prices[0]
    final_price = forecasted_prices[-1]
    percentage_change = ((final_price - initial_price) / initial_price) * 100

    if percentage_change > 0:
        trend = f"Bullish ({len(forecasted_prices)} days)"
        recommendation = "Buy"
    elif percentage_change < 0:
        trend = f"Bearish ({len(forecasted_prices)} days)"
        recommendation = "Sell"
    else:
        trend = f"Neutral ({len(forecasted_prices)} days)"
        recommendation = "Hold"

    return trend, recommendation
