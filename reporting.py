import os
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from unidecode import unidecode


def generate_pdf(
    ticker: str,
    data: pd.DataFrame,
    forecasted_prices,
    mae: float,
    mse: float,
    full_name: str,
    current_price: float,
    description: str,
    industry: str,
    trend: str,
    recommendation: str,
    news: List[dict],
    exchange: str,
) -> str:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=unidecode(f"Stock Forecast Report for {ticker}"), ln=True, align="C")
    pdf.cell(200, 10, txt=unidecode(f"Mean Absolute Error (MAE): {mae}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Mean Squared Error (MSE): {mse}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"{ticker} {full_name}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Current Price: {current_price}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Exchange: {exchange}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Tendency: {trend}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Recommendation: {recommendation}"), ln=True, align="L")
    pdf.cell(200, 10, txt=unidecode(f"Industry: {industry}"), ln=True, align="L")

    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data["Adj Close"], label="Historical Prices")
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1), periods=len(forecasted_prices), freq="B"
    )
    plt.plot(future_dates, forecasted_prices, label="Forecast")
    plt.title(f"{ticker} Stock Price Trend Forecast ({recommendation})")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()

    pdf.image("forecast.png", x=10, y=80, w=190)
    pdf.set_y(150)

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(200, 10, txt=unidecode("Description:"), ln=True, align="L")
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, unidecode(description))

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(200, 10, txt=unidecode("Latest News:"), ln=True, align="L")
    pdf.set_font("Arial", size=12)
    for article in news:
        pdf.cell(200, 10, txt=unidecode(article["title"]), ln=True, align="L")
        pdf.multi_cell(0, 10, unidecode(article["description"]))
        pdf.cell(200, 10, txt=unidecode(article["url"]), ln=True, align="L")

    output_dir = "reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf_path = os.path.join(output_dir, f"{ticker}_forecast_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

