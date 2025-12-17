# Stock Forecasting with LSTM

An end-to-end stock price forecasting application that uses technical indicators and an LSTM neural network to predict short-term market trends.

The project fetches historical market data, engineers financial indicators, trains a time-series model, and generates automated reports with forecasts, performance metrics, and related news.

## Key Features
- Historical stock and market index data ingestion (Yahoo Finance)
- Technical indicator engineering (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- LSTM-based time-series forecasting
- Trend classification (Bullish / Bearish / Neutral)
- Automated PDF report generation with charts and news
- Simple desktop interface for user interaction

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras (LSTM)
- yfinance
- Tkinter
- Matplotlib

## How It Works
1. User selects a stock ticker
2. Historical market data is fetched and processed
3. Technical indicators are calculated and scaled
4. An LSTM model is trained and evaluated
5. Future price trends are forecasted
6. A PDF report is generated with insights and recommendations

## Disclaimer
This project is for educational and research purposes only and does not constitute financial or investment advice.
