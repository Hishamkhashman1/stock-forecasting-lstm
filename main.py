import os
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from app import run_forecast


def main():
    root = tk.Tk()
    root.withdraw()

    ticker = simpledialog.askstring("Input", "Enter Stock Ticker:")
    if not ticker:
        messagebox.showerror("Error", "Please enter a stock ticker.")
        return

    progress_window = tk.Toplevel(root)
    progress_window.title("Forecasting in Progress")
    progress_label = tk.Label(progress_window, text="Forecasting the future in progress...")
    progress_label.pack(pady=10)
    progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
    progress_bar.pack(pady=10, padx=20)
    progress_bar.start()
    root.update()

    try:
        result = run_forecast(ticker)
        pdf_path = result["pdf_path"]
        messagebox.showinfo(
            "Info",
            f"Report generated for {ticker}\nLocation: {pdf_path}",
        )
        print(f"PDF report generated at: {pdf_path}")
        if hasattr(os, "startfile"):
            os.startfile(pdf_path)  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-except
        messagebox.showerror("Error", f"An error occurred: {exc}")
        print(f"An error occurred: {exc}")
    finally:
        progress_bar.stop()
        progress_window.destroy()
        root.destroy()


if __name__ == "__main__":
    main()
