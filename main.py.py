import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import threading
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define global variables
df = None
model = None
model_type = "RandomForest"

# Function to fetch real-time stock data
def fetch_stock_data(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        data = stock.history(period="1y")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

# Function to plot candlestick chart
def plot_candlestick_chart(stock_symbol):
    data = fetch_stock_data(stock_symbol)
    if data is None or data.empty:
        messagebox.showerror("Error", "Failed to fetch stock data.")
        return
    
    data.set_index("Date", inplace=True)
    mpf.plot(data, type='candle', style='charles', title=f"Candlestick Chart: {stock_symbol}", volume=True)

# Feature Engineering
def prepare_features(data):
    data['Close_Lag1'] = data['Close'].shift(1)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))
    data.dropna(inplace=True)
    return data

# Train ML Model
def train_model(stock_symbol):
    global model
    data = fetch_stock_data(stock_symbol)
    if data is None or data.empty:
        messagebox.showerror("Error", "Failed to fetch stock data.")
        return
    
    data = prepare_features(data)
    X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI']]
    y = data['Close']
    
    if len(X) < 2:
        messagebox.showerror("Error", "Not enough data to train the model.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    
    model.fit(X_train, y_train)
    joblib.dump(model, "stock_model.pkl")
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    messagebox.showinfo("Model Training", f"Model trained successfully!\nMean Absolute Error: {mae:.2f}")

# Predict future stock prices
def predict_future(stock_symbol):
    global model
    if model is None:
        messagebox.showerror("Error", "Please train the model first.")
        return
    
    data = fetch_stock_data(stock_symbol)
    if data is None or data.empty:
        return
    
    data = prepare_features(data)
    X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI']]
    
    predictions = model.predict(X)
    
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label="Actual Price", color='blue')
    plt.plot(data['Date'], predictions, label="Predicted Price", color='red', linestyle='dashed')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Predicted vs Actual Prices: {stock_symbol}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Create Tkinter GUI
root = tk.Tk()
root.title("Stock Prediction Tool")
root.geometry("600x400")

# UI Elements
stock_label = tk.Label(root, text="Enter Stock Symbol:")
stock_label.pack()
stock_entry = tk.Entry(root)
stock_entry.pack()

def set_model_rf():
    global model_type
    model_type = "RandomForest"

def set_model_xgb():
    global model_type
    model_type = "XGBoost"

model_label = tk.Label(root, text="Select Model:")
model_label.pack()
rf_button = ttk.Button(root, text="RandomForest", command=set_model_rf)
rf_button.pack()
xgb_button = ttk.Button(root, text="XGBoost", command=set_model_xgb)
xgb_button.pack()

fetch_button = ttk.Button(root, text="Fetch Data", command=lambda: fetch_stock_data(stock_entry.get()))
fetch_button.pack()

candlestick_button = ttk.Button(root, text="Candlestick Chart", command=lambda: plot_candlestick_chart(stock_entry.get()))
candlestick_button.pack()

train_button = ttk.Button(root, text="Train Model", command=lambda: train_model(stock_entry.get()))
train_button.pack()

predict_button = ttk.Button(root, text="Predict Future", command=lambda: predict_future(stock_entry.get()))
predict_button.pack()

root.mainloop()
