import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def calculate_technical_indicators(df):
    """
    Adds RSI and Moving Averages to the DataFrame.
    """
    df = df.copy()
    
    # RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # Moving Averages
    sma_50 = SMAIndicator(close=df['Close'], window=50)
    sma_200 = SMAIndicator(close=df['Close'], window=200)
    
    df['SMA_50'] = sma_50.sma_indicator()
    df['SMA_200'] = sma_200.sma_indicator()
    
    return df

def determine_trend(df):
    """
    Determines if the trend is Bullish or Bearish based on current indicators.
    """
    latest = df.iloc[-1]
    
    trend = "Neutral ⚪"
    if pd.isna(latest['SMA_50']) or pd.isna(latest['SMA_200']):
        return trend
        
    if latest['SMA_50'] > latest['SMA_200'] and latest['RSI'] > 50:
        trend = "Bullish 🟢"
    elif latest['SMA_50'] < latest['SMA_200'] and latest['RSI'] < 50:
        trend = "Bearish 🔴"
        
    return trend

def get_action_signal(current_price, predicted_price, trend, confidence):
    """
    Produces Buy/Sell signal based on custom logic.
    """
    if pd.isna(predicted_price):
        return "HOLD", 0.0
        
    price_diff = (predicted_price - current_price) / current_price * 100
    
    # Logic: If price is predicted to go up by at least 1% and trend is not completely bearish
    if predicted_price > current_price and "Bullish" in trend:
        if price_diff > 1.5:
            return "STRONG BUY ⭐", min(confidence + 10, 99.9)
        return "BUY 🟢", confidence
        
    elif predicted_price < current_price and "Bearish" in trend:
        if price_diff < -1.5:
            return "STRONG SELL 🚨", min(confidence + 10, 99.9)
        return "SELL 🔴", confidence
        
    # Overbought/Oversold logic
    if "Bullish" in trend and predicted_price < current_price:
         return "HOLD ⚪ (Possible Pullback)", confidence * 0.8
         
    return "HOLD ⚪", confidence * 0.7
