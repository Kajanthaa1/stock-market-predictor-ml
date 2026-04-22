import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
        
    def train_linear_regression(self, df, horizon=1):
        """
        Trains a simple Linear Regression model to predict future 'Close' price.
        """
        
        # Ensure required columns are present
        features = ['Prev_Close', 'MA_5', 'MA_10']
        if not all(col in df.columns for col in features):
            raise ValueError("Dataframe must contain 'Prev_Close', 'MA_5', and 'MA_10'")
        
        df = df.copy()
        df['Target'] = df['Close'].shift(-horizon)
        df = df.dropna(subset=['Target'] + features)
        
        X = df[features]
        y = df['Target']
        
        # Simple train-test split (80-20)
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.lr_model.fit(X_train, y_train)
        
        # Calculate accuracy/confidence score (R^2 roughly)
        score = self.lr_model.score(X_test, y_test)
        return score
        
    def predict_lr(self, current_features):
        """
        Predict next day close using LR.
        current_features should be a DataFrame with ['Prev_Close', 'MA_5', 'MA_10'] for the latest day
        """
        return self.lr_model.predict(current_features)[0]
        
    def prepare_lstm_data(self, df, look_back=60, horizon=1):
        """
        Prepares data for LSTM model (Time Series).
        """
        data = df.filter(['Close']).values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(look_back, len(scaled_data) - horizon + 1):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i + horizon - 1, 0])
           
        X, y = np.array(X), np.array(y)
        if len(X) > 0:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaled_data

    def train_lstm(self, df, epochs=3, batch_size=32, horizon=1):
        """
        Builds and trains an LSTM model.
        """
        look_back = 60
        X, y, scaled_data = self.prepare_lstm_data(df, look_back, horizon)
        
        if len(X) == 0:
            return 0.0 # Not enough data
         
        # Split data
        split = int(len(X) * 0.8)
        if split == 0:
            split = 1
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # Build model
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.lstm_model.add(LSTM(50, return_sequences=False))
        self.lstm_model.add(Dense(25))
        self.lstm_model.add(Dense(1))
        
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train
        if len(X_test) > 0:
            self.lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
            predictions = self.lstm_model.predict(X_test, verbose=0)
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        else:
            self.lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            rmse = 0.5 # Default penalty for small data
        
        # Heuristic confidence score based on RMSE and stock price std dev
        std_dev = np.std(scaled_data)
        confidence = max(0, min(100, 100 * (1 - (rmse / (std_dev + 1e-10)))))
        
        return confidence
        
    def predict_lstm(self, df):
        """
        Predict next day close using LSTM.
        """
        if self.lstm_model is None:
            return None
           
        look_back = 60
        data = df.filter(['Close']).values
        if len(data) < look_back:
            return data[-1][0] # Fallback
            
        last_60_days = data[-look_back:]
        last_60_days_scaled = self.scaler.transform(last_60_days)
        
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = self.lstm_model.predict(X_test, verbose=0)
        pred_price = self.scaler.inverse_transform(pred_price)
        
        return pred_price[0][0]
        