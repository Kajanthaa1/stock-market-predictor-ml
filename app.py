import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Ensure src module is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_handler import fetch_data, prepare_features_for_lr
from src.models import StockPredictor
from src.strategy import calculate_technical_indicators, determine_trend, get_action_signal

st.set_page_config(page_title="Stock Market Decision Engine", layout="wide", page_icon="📈")

st.title("📈 AI-Based Stock Market Decision Engine")
st.markdown("**Predict price • Detect trend • Recommend action**")

# Sidebar
st.sidebar.header("Configuration")
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "^GSPC"]
selected_ticker = st.sidebar.selectbox("Select Stock/Index", tickers)
period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y"], index=1)
model_choice = st.sidebar.radio("Select Prediction Model", ["Linear Regression (Fast)", "LSTM (Accurate but Slower)"])

st.sidebar.markdown("---")
st.sidebar.info("Note: Predictions are based on historical data. Not to be used as financial advice. Market holds inherent risks.")

# Fetch Data
st.write(f"### Current Analytics for **{selected_ticker}**")
with st.spinner('Fetching Real-time Market Data...'):
    df_raw = fetch_data(selected_ticker, period)

if df_raw is None or len(df_raw) < 200:
    st.error("Failed to fetch enough data. Please try another stock or check your internet connection.")
else:
    df_indicators = calculate_technical_indicators(df_raw)
    current_price = df_indicators.iloc[-1]['Close']
    current_date = df_indicators.iloc[-1]['Date']
    
    # Process Modeling
    predictor = StockPredictor()
    predicted_price = None
    confidence = 0.0
    
    with st.spinner(f"Training {model_choice} model based on {selected_ticker} history..."):
        if "Linear" in model_choice:
            df_lr = prepare_features_for_lr(df_raw)
            lr_score = predictor.train_linear_regression(df_lr)
            confidence = min(max(lr_score * 100, 20), 99.9) # Bound between 20 and 99.9%
            
            # Predict next based on last available features
            latest_features = df_lr[['Prev_Close', 'MA_5', 'MA_10']].iloc[-1:]
            predicted_price = predictor.predict_lr(latest_features)
            
        elif "LSTM" in model_choice:
            df_clean = df_raw.dropna(subset=['Close'])
            confidence = predictor.train_lstm(df_clean, epochs=3)
            predicted_price = predictor.predict_lstm(df_clean)

    # Trend and Actions logic processing
    trend = determine_trend(df_indicators)
    action, signal_conf = get_action_signal(current_price, predicted_price, trend, confidence)
    
    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${current_price:.2f}")
    
    if predicted_price is not None:
        delta = predicted_price - current_price
        col2.metric("Predicted Next Day", f"${predicted_price:.2f}", f"{delta:.2f} (Prediction)")
    
    col3.metric("Current Trend", trend)
    col4.metric("Action Recommendation", action, f"{signal_conf:.1f}% Model Confidence")
    
    st.markdown("---")
    
    # Charting with Plotly
    st.write("### Interactive Price Chart & Technical Indicators")
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df_indicators['Date'],
                open=df_indicators['Open'],
                high=df_indicators['High'],
                low=df_indicators['Low'],
                close=df_indicators['Close'],
                name='Candlesticks'))
                
    # Moving Averages
    fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['SMA_50'], line=dict(color='orange', width=2), name='50-Day SMA (Short Trend)'))
    fig.add_trace(go.Scatter(x=df_indicators['Date'], y=df_indicators['SMA_200'], line=dict(color='deepskyblue', width=2), name='200-Day SMA (Long Trend)'))
    
    # Predicted Future Point marker
    if predicted_price is not None:
        next_day = current_date + pd.Timedelta(days=1)
        fig.add_trace(go.Scatter(x=[next_day], y=[predicted_price], mode='markers', marker=dict(color='magenta', size=12, symbol='star'), name='Predicted Point'))
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark", 
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    # Expandable raw data section
    with st.expander("View Underlying Data & Indicators"):
        st.dataframe(df_indicators[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_50', 'SMA_200']].tail(20))
