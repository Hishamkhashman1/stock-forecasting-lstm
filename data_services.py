import datetime
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from sklearn.preprocessing import StandardScaler


DEFAULT_NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "ab57665b93dc4be5be77fdf1154cf384")
newsapi = NewsApiClient(api_key=DEFAULT_NEWS_API_KEY)


def fetch_data(ticker: str, months: int = 6) -> pd.DataFrame:
    return _fetch_time_series(ticker, months, "stock")


def fetch_exchange_data(exchange_ticker: str, months: int = 6) -> pd.DataFrame:
    return _fetch_time_series(exchange_ticker, months, "exchange")


def _fetch_time_series(ticker: str, months: int, label: str) -> pd.DataFrame:
    end = datetime.datetime.now() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=months * 30)
    stock_data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
    )
    if stock_data.empty:
        raise ValueError(f"No data found for {label} ticker {ticker}")

    if "Adj Close" not in stock_data.columns:
        stock_data["Adj Close"] = stock_data["Open"].shift(-1)
        stock_data = stock_data.dropna()

    print(f"Fetched {len(stock_data)} samples for {label} ticker {ticker}")
    return stock_data


def fetch_company_info(ticker: str) -> Tuple[str, float, str, str, str]:
    stock = yf.Ticker(ticker)
    info = stock.info
    full_name = info.get("longName", "N/A")
    current_price = info.get("currentPrice", "N/A")
    description = info.get("longBusinessSummary", "N/A")
    industry = info.get("industry", "N/A")
    exchange = info.get("exchange", "N/A")
    return full_name, current_price, description, industry, exchange


def fetch_latest_news(ticker: str) -> List[dict]:
    articles = newsapi.get_everything(
        q=ticker, language="en", sort_by="publishedAt", page_size=5
    )
    news = []
    for article in articles["articles"]:
        news.append(
            {
                "title": article["title"] if article["title"] else "No title available",
                "description": article["description"]
                if article["description"]
                else "No description available",
                "url": article["url"] if article["url"] else "No URL available",
            }
        )
    return news


def calculate_technical_indicators(stock_data: pd.DataFrame) -> pd.DataFrame:
    stock_data = stock_data.copy()
    stock_data["SMA"] = stock_data["Adj Close"].rolling(window=20).mean()
    stock_data["EMA"] = stock_data["Adj Close"].ewm(span=20, adjust=False).mean()
    pct_change = stock_data["Adj Close"].pct_change()
    rsi_numerator = pct_change.rolling(window=14).mean()
    rsi_denominator = pct_change.rolling(window=14).std()
    stock_data["RSI"] = 100 - (100 / (1 + rsi_numerator / rsi_denominator))
    stock_data["MACD"] = stock_data["Adj Close"].ewm(span=12, adjust=False).mean() - stock_data[
        "Adj Close"
    ].ewm(span=26, adjust=False).mean()
    stock_data["MACD_Signal"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()
    stock_data["MACD_Histogram"] = stock_data["MACD"] - stock_data["MACD_Signal"]
    rolling_std = stock_data["Adj Close"].rolling(window=20).std()
    stock_data["Upper_BB"] = stock_data["SMA"] + (rolling_std * 2)
    stock_data["Lower_BB"] = stock_data["SMA"] - (rolling_std * 2)
    stock_data["ATR"] = stock_data["High"].rolling(window=14).max() - stock_data["Low"].rolling(
        window=14
    ).min()
    stock_data["CCI"] = (
        stock_data["Adj Close"] - stock_data["Adj Close"].rolling(window=20).mean()
    ) / (0.015 * rolling_std)
    stock_data["Momentum"] = stock_data["Adj Close"] - stock_data["Adj Close"].shift(14)
    stock_data["Parabolic_SAR"] = stock_data["Adj Close"].ewm(span=20, adjust=False).mean()
    stock_data["Volume"] = stock_data["Volume"]
    return stock_data


def preprocess_data(
    stock_data: pd.DataFrame,
    exchange_data: pd.DataFrame,
    sequence_length: int = 120,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    stock_data = calculate_technical_indicators(stock_data)
    exchange_data = calculate_technical_indicators(exchange_data)

    stock_data = stock_data.dropna()
    exchange_data = exchange_data.dropna()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

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

    combined_data = pd.concat([stock_data, exchange_data], axis=1)

    scaled_features = feature_scaler.fit_transform(combined_data.values)
    scaled_target = target_scaler.fit_transform(stock_data[["Adj Close"]].values)

    X, y = [], []
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i - sequence_length : i])
        y.append(scaled_target[i, 0])

    X, y = np.array(X), np.array(y)
    print(f"Preprocessed data size: {X.shape[0]} samples")
    return X, y, feature_scaler, target_scaler

