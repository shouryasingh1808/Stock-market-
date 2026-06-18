import yfinance as yf
import pandas as pd

def fetch_stock_data(period = "2y"):
    data = yf.download("^NSEI" , period=period)

    data.dropna(inplace = True)

    return data

nifty_data = fetch_stock_data()
print(nifty_data.head())