import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from data import fetch_stock_data, add_features

def train_model(stock_name= "^NSEI"):
    # Get the data--
    data = fetch_stock_data(stock_name)
    data = add_features(data)
    
    # Define feature and target
    features = ["MA_20", "MA_50", "Yesterday", "Last_week", "RSI"]
    target = "Close"

    X = data[features]
    Y = data[target]

    X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , shuffle=False)
    