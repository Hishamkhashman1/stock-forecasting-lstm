from typing import Any, Dict

from sklearn.model_selection import train_test_split

from data_services import (
    fetch_company_info,
    fetch_data,
    fetch_exchange_data,
    fetch_latest_news,
    preprocess_data,
)
from model_services import (
    determine_trend_and_recommendation,
    forecast_trend,
    train_lstm_model,
)
from reporting import generate_pdf


def run_forecast(
    ticker: str,
    *,
    exchange_ticker: str = "^GSPC",
    training_attempts: int = 5,
    min_test_split_samples: int = 5,
) -> Dict[str, Any]:
    data = fetch_data(ticker)
    full_name, current_price, description, industry, exchange = fetch_company_info(ticker)
    exchange_data = fetch_exchange_data(exchange_ticker)

    X, y, feature_scaler, target_scaler = preprocess_data(data, exchange_data)
    if len(X) == 0:
        raise ValueError(
            "Not enough data to create sequences. Try increasing the date range or reducing the sequence length."
        )

    best_model = None
    best_mae = float("inf")
    best_mse = None

    for attempt in range(training_attempts):
        print(f"Training attempt {attempt + 1} of {training_attempts}")
        test_size = 0.5 if len(X) < min_test_split_samples else 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=attempt
        )
        model, mae, mse = train_lstm_model(X_train, y_train, X_test, y_test, target_scaler)
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_mse = mse

    if best_model is None:
        raise RuntimeError("Model training failed on all attempts.")

    print(f"Best MAE: {best_mae}")

    forecasted_prices = forecast_trend(
        best_model, data, exchange_data, feature_scaler, target_scaler
    )
    trend, recommendation = determine_trend_and_recommendation(forecasted_prices)
    news = fetch_latest_news(ticker)

    pdf_path = generate_pdf(
        ticker,
        data,
        forecasted_prices,
        best_mae,
        best_mse,
        full_name,
        current_price,
        description,
        industry,
        trend,
        recommendation,
        news,
        exchange,
    )

    return {
        "pdf_path": pdf_path,
        "trend": trend,
        "recommendation": recommendation,
        "mae": best_mae,
        "mse": best_mse,
    }
