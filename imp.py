import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

# Step 1: Download and Prepare Data
def get_stock_data(ticker="SBIN.NS", period="1y", interval="1d"):
    """Download stock data and save locally"""
    df = yf.download(ticker, period=period, interval=interval)
    os.makedirs("data", exist_ok=True)
    df.reset_index(inplace=True)
    df.to_csv(f"data/{ticker.split('.')[0].lower()}.csv", index=False)
    return df

# Step 2: Feature Engineering
def create_features(df):
    """Create technical indicators and target variable"""
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Technical Indicators
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    
    # Target: 1 if next day's close > today, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Step 3: Train Model
def train_model(X, y):
    """Train Random Forest classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
    
    return model, scaler

# Step 4: Predict Future
def predict_future(model, scaler, df, features, days_to_predict=5):
    """Predict future price movements"""
    # Prepare last available data point
    last_data = df[features].iloc[-1:].copy()
    
    predictions = []
    future_dates = []
    current_date = df.index[-1]
    
    for _ in range(days_to_predict):
        # Scale features
        scaled_features = scaler.transform(last_data)
        
        # Predict
        pred = model.predict(scaled_features)[0]
        predictions.append(pred)
        
        # Update date
        current_date += timedelta(days=1)
        future_dates.append(current_date)
        
        # Update features for next prediction (simplified approach)
        new_row = last_data.copy()
        if pred == 1:
            new_row['Close_Lag1'] = new_row['Close_Lag1'] * 1.01  # Assume 1% increase
        else:
            new_row['Close_Lag1'] = new_row['Close_Lag1'] * 0.99  # Assume 1% decrease
            
        new_row['Close_Lag2'] = last_data['Close_Lag1'].values[0]
        new_row['Daily_Return'] = (new_row['Close_Lag1'] - new_row['Close_Lag2']) / new_row['Close_Lag2']
        
        # Update moving averages (simplified)
        new_row['SMA_10'] = (last_data['SMA_10'] * 10 - last_data['Close_Lag2'] + new_row['Close_Lag1']) / 10
        new_row['SMA_20'] = (last_data['SMA_20'] * 20 - last_data['Close_Lag2'] + new_row['Close_Lag1']) / 20
        
        last_data = new_row
    
    return future_dates, predictions

# Step 5: Visualization
def plot_results(df, future_dates, predictions):
    """Visualize historical and predicted data"""
    plt.figure(figsize=(14, 7))
    
    # Historical Close Price
    plt.plot(df.index, df['Close'], label='Historical Close', color='blue')
    
    # Future Predictions
    future_dates = pd.to_datetime(future_dates)
    colors = ['green' if pred == 1 else 'red' for pred in predictions]
    for date, color in zip(future_dates, colors):
        plt.axvline(x=date, color=color, alpha=0.3, linestyle='--')
    
    plt.title('Stock Price with Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Get data
    ticker = "SBIN.NS"
    df = get_stock_data(ticker)
    df = pd.read_csv(f"data/{ticker.split('.')[0].lower()}.csv", parse_dates=['Date'], index_col='Date')
    
    # Feature engineering
    df = create_features(df)
    features = ['SMA_10', 'SMA_20', 'Daily_Return', 'Close_Lag1', 'Close_Lag2', 'MACD', 'RSI']
    X = df[features]
    y = df['Target']
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Predict future
    future_days = 5
    future_dates, predictions = predict_future(model, scaler, df, features, future_days)
    
    print("\nFuture Predictions:")
    for date, pred in zip(future_dates, predictions):
        print(f"{date.strftime('%Y-%m-%d')}: {'Up' if pred == 1 else 'Down'}")
    
    # Visualize
    plot_results(df, future_dates, predictions)