import pandas as pd
import os
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from data import fetch_stock_data, add_features

def train_model(stock_name= "^NSEI"):
    if os.path.exists("model.pkl"):
        print("Model already exists! Loading starts..")
        return joblib.load("model.pkl")
    # Get the data--
    data = fetch_stock_data(stock_name)
    data = add_features(data)

    data["Next open"] = data["Open"].shift(-1)
    data.dropna(inplace=True)

    # Define feature and target
    features = ["MA_20", "MA_50", "Yesterday", "Last_week", "RSI"]
    target = "Close"

    X = data[features]
    Y = data[target]

    X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , shuffle=False)

    # Craetes pipeline --
    pipeline = Pipeline([
        ("scaler" , StandardScaler()),
        ("model"  , RandomForestRegressor(random_state=42 , n_estimators=100))
    ])
    
    pipeline.fit(X_train , Y_train)

    prediction = pipeline.predict(X_test)
    rmse = mean_squared_error(Y_test,prediction) ** 0.5
    print(f"RMSE: {rmse:.2f}")
     
    # Save the pipeline--

    joblib.dump(pipeline, "model.pkl")
    print("model saved!")

    return pipeline

train_model()