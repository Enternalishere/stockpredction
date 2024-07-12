import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Define global variables
df = None
model = None

def load_nifty_data():
    global df
    try:
        filename = "C:/Users/HP/Desktop/stock prediction/NIFTY NEXT 50-16-03-2023-to-16-03-2024.csv"
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()
        messagebox.showinfo("Success", "Nifty data loaded successfully!")
        print(df.head())  # Print the first few rows of the DataFrame
    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_close_graph():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load data first.")
        return

    try:
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.title('Historical Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
        plt.xticks(rotation=45) 
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_open_close_graph():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load data first.")
        return

    try:
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Open'], label='Open Price', color='green')
        plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
        plt.title('Historical Open and Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
        plt.xticks(rotation=45) 
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_low_high_graph():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load data first.")
        return

    try:
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Low'], label='Low Price', color='red')
        plt.plot(df['Date'], df['High'], label='High Price', color='orange')
        plt.title('Historical Low and High Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
        plt.xticks(rotation=45) 
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def train_model():
    global df, model
    if df is None:
        messagebox.showerror("Error", "Please load data first.")
        return

    try:
        # Create a new column with lagged closing prices for prediction
        df['Close_lag1'] = df['Close'].shift(1)
        
        # Drop any rows with NaN values that were created due to shifting
        df.dropna(inplace=True)
        
        # Define the features and target variable for the model
        features = df[['Close_lag1']]  # Use lagged 'Close' column as the feature
        target = df['Close']
        
        X_train, _, y_train, _ = train_test_split(features, target, test_size=0.2, random_state=42, shuffle=False)
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        messagebox.showinfo("Success", "Model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def calculate_accuracy():
    global df, model
    if df is None or model is None:
        messagebox.showerror("Error", "Please load data and train model first.")
        return

    try:
        # Prepare data for prediction
        df['Close_lag1'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        features = df[['Close_lag1']]
        target = df['Close']

        # Make predictions
        predictions = model.predict(features)

        # Calculate accuracy
        r_squared = r2_score(target, predictions)
        mae = mean_absolute_error(target, predictions)

        messagebox.showinfo("Accuracy", f"R-squared Score: {r_squared:.2f}\nMean Absolute Error: {mae:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_predicted_vs_actual():
    global df, model
    if df is None or model is None:
        messagebox.showerror("Error", "Please load data and train model first.")
        return

    try:
        # Prepare data for prediction
        df['Close_lag1'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        features = df[['Close_lag1']]
        target = df['Close']

        # Make predictions
        predictions = model.predict(features)

        # Plot actual vs. predicted closing prices
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], target, label='Actual Closing Prices', color='blue')
        plt.plot(df['Date'], predictions, label='Predicted Closing Prices', color='red', linestyle='--')
        plt.title('Actual vs. Predicted Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
        plt.xticks(rotation=45) 
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_predicted_data_table():
    global df, model
    if df is None or model is None:
        messagebox.showerror("Error", "Please load data and train model first.")
        return

    try:
        # Prepare data for prediction
        df['Close_lag1'] = df['Close'].shift(1)
        df.dropna(inplace=True)
        features = df[['Close_lag1']]

        # Make predictions
        df['Predicted_Close'] = model.predict(features)

        # Display predicted data in a table
        top = tk.Toplevel()
        top.title("Predicted Closing Prices")

        tree = ttk.Treeview(top)
        tree["columns"] = ("Date", "Predicted Close")
        tree.heading("#0", text="Index")
        tree.heading("Date", text="Date")
        tree.heading("Predicted Close", text="Predicted Close")
        
        for index, row in df.iterrows():
            tree.insert("", "end", text=index, values=(row['Date'], row['Predicted_Close']))

        tree.pack(expand=True, fill="both")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    root.title("Data Analysis Tool")

    # Load Nifty Data button
    nifty_button = ttk.Button(root, text="Load Nifty Data", command=load_nifty_data)
    nifty_button.pack(pady=5)

    # Train Model button
    train_button = ttk.Button(root, text="Train Model", command=train_model)
    train_button.pack(pady=5)

    # Plot Close Graph button
    close_button = ttk.Button(root, text="Plot Close Graph", command=plot_close_graph)
    close_button.pack(pady=5)

    # Plot Open-Close Graph button
    open_close_button = ttk.Button(root, text="Plot Open-Close Graph", command=plot_open_close_graph)
    open_close_button.pack(pady=5)

    # Plot Low-High Graph button
    low_high_button = ttk.Button(root, text="Plot Low-High Graph", command=plot_low_high_graph)
    low_high_button.pack(pady=5)

    # Calculate Accuracy button
    accuracy_button = ttk.Button(root, text="Calculate Accuracy", command=calculate_accuracy)
    accuracy_button.pack(pady=5)

    # Plot Predicted vs. Actual Graph button
    predicted_vs_actual_button = ttk.Button(root, text="Plot Predicted vs. Actual Graph", command=plot_predicted_vs_actual)
    predicted_vs_actual_button.pack(pady=5)

    # Show Predicted Data Table button
    predicted_data_button = ttk.Button(root, text="Show Predicted Data Table", command=show_predicted_data_table)
    predicted_data_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
