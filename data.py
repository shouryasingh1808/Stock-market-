import yfinance as yf
import pandas as pd

def fetch_stock_data(stock_name, period = "2y"):
    data = yf.download(stock_name , period=period)

    data.dropna(inplace = True)

    return data

def add_features(data):
    # Moving average
    data["MA_20"] = data["Close"].rolling(20).mean() # Average price of last 20 days
    data["MA_50"] = data["Close"].rolling(50).mean() # Average price of last 50 days

    # Past price
    data["Yesterday"] = data["Close"].shift(1)   # The yesterday price
    data["Last_week"] = data["Close"].shift(7)   # The price 7 days ago

    # RSI Calculation
    # The price changed each day
    change = data["Close"].diff()

    # Average gain on days price went up--
    avg_gain = change.where(change > 0 ,0 ).rolling(14).mean()
    # Average gain on days price went down--
    avg_loss = -change.where(change < 0 ,0 ).rolling(14).mean()

    # RSI score
    rs = avg_gain/avg_loss
    data["RSI"] = 100 - (100/(1+rs))

    data.dropna(inplace = True)
    
    return data


