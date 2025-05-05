import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import re
from datetime import datetime, timedelta
from matplotlib.widgets import Slider

class StockPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Prediction Tool")
        self.root.geometry("800x600")
        self.root.configure(bg="#1E1E2F")
        self.models = {}  # Dictionary to store models by stock symbol
        self.df = None
        self.setup_gui()

    def is_valid_symbol(self, symbol):
        return bool(re.match(r'^[A-Z.-]+$', symbol.strip().upper()))

    def fetch_stock_data(self, stock_symbol, period="1y"):
        # Handle multiple symbols by taking the first valid one
        if not stock_symbol or not any(self.is_valid_symbol(s.strip()) for s in stock_symbol.split(',')):
            messagebox.showerror("Error", "Please enter at least one valid stock symbol (e.g., AAPL, MSFT).")
            return None
        symbols = [s.strip() for s in stock_symbol.split(',') if self.is_valid_symbol(s.strip())]
        if not symbols:
            messagebox.showerror("Error", "No valid stock symbols provided.")
            return None
        symbol = symbols[0]  # Use the first valid symbol
        if len(symbols) > 1:
            messagebox.showwarning("Warning", f"Using only the first symbol: {symbol}. Use 'Train Multiple' for multiple symbols.")
        try:
            self.status_label.config(text=f"Fetching data for {symbol}...")
            self.progress_bar.start()
            self.root.update()
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                messagebox.showerror("Error", f"No data found for {symbol}. Check the symbol or try again.")
                return None
            data.reset_index(inplace=True)
            self.df = data
            self.status_label.config(text=f"Data fetched for {symbol} successfully!")
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")
            return None
        finally:
            self.progress_bar.stop()
            self.root.update()

    def calculate_rsi(self, data, periods=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_macd(self, data):
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data

    def calculate_vwap(self, data):
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return data

    def prepare_features(self, data, stock_symbol):
        if len(data) < 15:
            messagebox.showerror("Error", f"Not enough data for {stock_symbol}.")
            return None
        data['Close_Lag1'] = data['Close'].shift(1)
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data = self.calculate_rsi(data)
        data = self.calculate_macd(data)
        data = self.calculate_vwap(data)
        data.dropna(inplace=True)
        return data

    def plot_candlestick_chart(self, stock_symbol):
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
        data.set_index("Date", inplace=True)
        data = self.calculate_rsi(data)
        data = self.calculate_macd(data)
        data = self.calculate_vwap(data)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Upper_BB'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
        data['Lower_BB'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)

        apds = [
            mpf.make_addplot(data['SMA_20'], color='blue', label='SMA 20'),
            mpf.make_addplot(data['SMA_50'], color='red', label='SMA 50'),
            mpf.make_addplot(data['Upper_BB'], color='green', linestyle='dashed', label='Upper BB'),
            mpf.make_addplot(data['Lower_BB'], color='green', linestyle='dashed', label='Lower BB'),
            mpf.make_addplot(data['VWAP'], color='purple', label='VWAP'),
            mpf.make_addplot(data['RSI'], panel=1, color='orange', ylabel='RSI'),
            mpf.make_addplot(data['MACD'], panel=2, color='blue', ylabel='MACD'),
            mpf.make_addplot(data['Signal'], panel=2, color='red')
        ]

        mpf.plot(data, type='candle', style='yahoo', title=f"Candlestick Chart: {stock_symbol}",
                 volume=True, mav=(5, 10), addplot=apds, panel_ratios=(3, 1, 1), figscale=1.5)

    def train_model(self, stock_symbol):
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
        data = self.prepare_features(data, stock_symbol)
        if data is None:
            return
        X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
        y = data['Close']
        if len(X) < 2:
            messagebox.showerror("Error", "Not enough data to train the model.")
            return
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models[stock_symbol] = model
        joblib.dump(model, f"stock_model_{stock_symbol}.pkl")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        self.result_text.insert(tk.END, f"{stock_symbol} - MAE: {mae:.2f}, R²: {r2:.2f}\n")
        messagebox.showinfo("Model Training", f"Model for {stock_symbol} trained!\nMAE: {mae:.2f}, R²: {r2:.2f}")
        self.update_trained_models()

    def train_multiple_models(self):
        symbols = self.stock_entry.get().replace(" ", "").split(",")
        valid_symbols = [s for s in symbols if self.is_valid_symbol(s)]
        if not valid_symbols:
            messagebox.showerror("Error", "No valid stock symbols provided.")
            return
        for symbol in valid_symbols:
            self.train_model(symbol)

    def predict_future(self, stock_symbol, days_ahead=5):
        if stock_symbol not in self.models:
            messagebox.showerror("Error", f"No model trained for {stock_symbol}. Select from trained models.")
            return
        model = self.models[stock_symbol]
        data = self.fetch_stock_data(stock_symbol, self.period_var.get())
        if data is None or data.empty:
            return
        data = self.prepare_features(data, stock_symbol)
        if data is None:
            return
        X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
        predictions = model.predict(X)
        confidence_intervals = []
        for _ in range(100):
            bootstrap_model = RandomForestRegressor(n_estimators=10, random_state=np.random.randint(0, 1000))
            bootstrap_model.fit(X, data['Close'])
            confidence_intervals.append(bootstrap_model.predict(X))
        ci_lower = np.percentile(confidence_intervals, 5, axis=0)
        ci_upper = np.percentile(confidence_intervals, 95, axis=0)

        last_features = X.iloc[-1:].copy()
        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=days_ahead + 1, freq='B')[1:]
        future_predictions = []
        for _ in range(days_ahead):
            pred = model.predict(last_features)[0]
            future_predictions.append(pred)
            last_features['Close_Lag1'] = pred
            last_features['SMA_5'] = np.mean([pred] + last_features['Close_Lag1'].tolist()[-4:])
            last_features['SMA_10'] = np.mean([pred] + last_features['Close_Lag1'].tolist()[-9:])
            last_features['RSI'] = last_features['RSI'].iloc[0]
            last_features['MACD'] = last_features['MACD'].iloc[0]
            last_features['VWAP'] = last_features['VWAP'].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Date'], data['Close'], label="Actual Price", color='blue')
        ax.plot(data['Date'], predictions, label="Predicted Price", color='orange', linestyle='dashed')
        ax.fill_between(data['Date'], ci_lower, ci_upper, color='gray', alpha=0.3, label="90% Confidence Interval")
        ax.plot(future_dates, future_predictions, label="Future Prediction", color='red', linestyle='dashed')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Price Prediction: {stock_symbol}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def compare_stocks(self):
        symbols = self.stock_entry.get().replace(" ", "").split(",")
        valid_symbols = [s for s in symbols if s in self.models]
        if not valid_symbols:
            messagebox.showerror("Error", "No trained models for the entered symbols.")
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        for symbol in valid_symbols:
            data = self.fetch_stock_data(symbol, self.period_var.get())
            if data is None or data.empty:
                continue
            data = self.prepare_features(data, symbol)
            if data is None:
                continue
            X = data[['Close_Lag1', 'SMA_5', 'SMA_10', 'RSI', 'MACD', 'VWAP']]
            predictions = self.models[symbol].predict(X)
            ax.plot(data['Date'], predictions, label=f"{symbol} Predicted", linestyle='dashed')
            ax.plot(data['Date'], data['Close'], label=f"{symbol} Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title("Stock Price Comparison")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_in_thread(self, func, *args):
        def wrapper():
            self.status_label.config(text="Processing...")
            self.progress_bar.start()
            self.root.update()
            try:
                func(*args)
            finally:
                self.status_label.config(text="Ready")
                self.progress_bar.stop()
                self.root.update()
        threading.Thread(target=wrapper, daemon=True).start()

    def update_trained_models(self):
        self.model_combo['values'] = list(self.models.keys())

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Arial", 12), padding=6, background="#00ADB5", foreground="#000000")
        style.configure("TLabel", font=("Arial", 12), background="#1E1E2F", foreground="#FFFFFF")
        style.configure("TEntry", font=("Arial", 12))
        style.configure("TCombobox", font=("Arial", 12))

        main_frame = tk.Frame(self.root, bg="#1E1E2F")
        main_frame.pack(pady=20, padx=20, fill='both', expand=True)

        input_frame = tk.Frame(main_frame, bg="#1E1E2F")
        input_frame.pack(fill='x')

        tk.Label(input_frame, text="Stock Symbol(s):", font=("Arial", 12), bg="#1E1E2F", fg="#FFFFFF").grid(row=0, column=0, padx=5, pady=5)
        self.stock_entry = ttk.Entry(input_frame, font=("Arial", 12))
        self.stock_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.stock_entry.insert(0, "AAPL")

        tk.Label(input_frame, text="Period:", font=("Arial", 12), bg="#1E1E2F", fg="#FFFFFF").grid(row=0, column=2, padx=5, pady=5)
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var, values=["6mo", "1y", "2y"], state="readonly")
        period_combo.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(input_frame, text="Trained Models:", font=("Arial", 12), bg="#1E1E2F", fg="#FFFFFF").grid(row=1, column=0, padx=5, pady=5)
        self.model_combo = ttk.Combobox(input_frame, state="readonly")
        self.model_combo.grid(row=1, column=1, padx=5, pady=5, sticky='ew')

        button_frame = tk.Frame(main_frame, bg="#1E1E2F")
        button_frame.pack(pady=10, fill='x')

        buttons = [
            ("Fetch Data", lambda: self.run_in_thread(self.fetch_stock_data, self.stock_entry.get(), self.period_var.get())),
            ("Candlestick Chart", lambda: self.run_in_thread(self.plot_candlestick_chart, self.stock_entry.get())),
            ("Train Model", lambda: self.run_in_thread(self.train_model, self.stock_entry.get())),
            ("Train Multiple", lambda: self.run_in_thread(self.train_multiple_models)),
            ("Predict Future", lambda: self.run_in_thread(self.predict_future, self.stock_entry.get())),
            ("Compare Stocks", lambda: self.run_in_thread(self.compare_stocks))
        ]
        for i, (text, cmd) in enumerate(buttons):
            ttk.Button(button_frame, text=text, command=cmd).grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky='ew')

        self.status_label = ttk.Label(main_frame, text="Ready", font=("Arial", 10), background="#1E1E2F", foreground="#FFFFFF")
        self.status_label.pack(pady=5)

        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', padx=20)

        result_frame = tk.Frame(main_frame, bg="#1E1E2F")
        result_frame.pack(pady=10, fill='both', expand=True)
        tk.Label(result_frame, text="Model Results:", font=("Arial", 12), bg="#1E1E2F", fg="#FFFFFF").pack(anchor='w')
        self.result_text = tk.Text(result_frame, height=5, font=("Arial", 10), bg="#2D2D44", fg="#FFFFFF")
        self.result_text.pack(fill='both', padx=5, pady=5)

if __name__ == "__main__":
    # Ensure all required libraries are installed
    try:
        root = tk.Tk()
        app = StockPredictor(root)
        root.mainloop()
    except tk.TclError as e:
        print(f"Tkinter Error: {e}")
        print("If running in a headless environment, try using 'xvfb-run python stock_prediction_tool.py'")
        print("On Linux, install Xvfb with: sudo apt-get install xvfb")
        print("Ensure all dependencies are installed: pip install tkinter pandas numpy matplotlib mplfinance yfinance joblib scikit-learn seaborn")
