import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Ensure src module is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_handler import fetch_data, prepare_features_for_lr, fetch_news_sentiment, fetch_realtime_price
from src.models import StockPredictor
from src.strategy import calculate_technical_indicators, determine_trend, get_action_signal
from src.portfolio import generate_portfolio_suggestion
from src.alerts import check_and_trigger_alerts
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Stock Market Decision Engine", layout="wide", page_icon="📈")

# Auto-refresh cycle (every 30 seconds)
st_autorefresh(interval=30000, limit=1000, key="price_updater")

st.title("📈 AI-Based Stock Market Decision Engine")
st.markdown("**Predict price • Detect trend • Recommend action • Advanced Analytics**")

# Sidebar
st.sidebar.header("Configuration")
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "^GSPC"]
selected_ticker = st.sidebar.selectbox("Select Stock/Index", tickers)
period = st.sidebar.selectbox("Historical Data Period", ["1y", "2y", "5y"], index=1)
horizon_choice = st.sidebar.selectbox("Prediction Horizon", ["Next Day", "Next Week"])
horizon_days = 1 if horizon_choice == "Next Day" else 5
model_choice = st.sidebar.radio("Select Prediction Model", ["Linear Regression (Fast)", "LSTM (Accurate but Slower)"])

st.sidebar.markdown("---")
st.sidebar.info("Note: Predictions are based on historical data. Not to be used as financial advice. Market holds inherent risks.")

# Real-time price fetch & Alert Checking
realtime_price = fetch_realtime_price(selected_ticker)
if realtime_price:
    check_and_trigger_alerts(realtime_price, selected_ticker)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📰 News & Sentiment", "🔔 Alerts", "💼 Portfolio"])

with tab1:
    # Fetch Data
    st.write(f"### Current Analytics for **{selected_ticker}**")
    
    if realtime_price:
        st.markdown(f"**🔴 Live Price:** ${realtime_price:.2f}")

    with st.spinner('Fetching Historical Market Data...'):
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
                lr_score = predictor.train_linear_regression(df_lr, horizon=horizon_days)
                confidence = min(max(lr_score * 100, 20), 99.9) # Bound between 20 and 99.9%
                
                # Predict next based on last available features
                latest_features = df_lr[['Prev_Close', 'MA_5', 'MA_10']].iloc[-1:]
                predicted_price = predictor.predict_lr(latest_features)
                
            elif "LSTM" in model_choice:
                df_clean = df_raw.dropna(subset=['Close'])
                confidence = predictor.train_lstm(df_clean, epochs=3, horizon=horizon_days)
                predicted_price = predictor.predict_lstm(df_clean)

        # Trend and Actions logic processing
        trend = determine_trend(df_indicators)
        action, signal_conf = get_action_signal(current_price, predicted_price, trend, confidence)
        
        # Metric Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Close", f"${current_price:.2f}")
        
        if predicted_price is not None:
            delta = predicted_price - current_price
            col2.metric(f"Predicted {horizon_choice}", f"${predicted_price:.2f}", f"{delta:.2f} (Prediction)")
        
        col3.metric("Current Trend", trend)
        action_str = f"{action} ({int(signal_conf)}% confidence)"
        col4.metric("Action Recommendation", action_str)
        
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
            future_date = pd.bdate_range(current_date, periods=horizon_days+1)[-1]
            fig.add_trace(go.Scatter(x=[future_date], y=[predicted_price], mode='markers', marker=dict(color='magenta', size=12, symbol='star'), name=f'Predicted {horizon_choice}'))
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark", 
                          margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Expandable raw data section
        with st.expander("View Underlying Data & Indicators"):
            st.dataframe(df_indicators[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'SMA_50', 'SMA_200']].tail(20))

with tab2:
    st.write(f"### News & NLP Sentiment for {selected_ticker}")
    with st.spinner("Analyzing recent news headlines..."):
        avg_score, news_list = fetch_news_sentiment(selected_ticker)
        
        if news_list:
            if avg_score > 0.1:
                st.success(f"Overall Market Sentiment: **Bullish** (Score: {avg_score:.2f})")
            elif avg_score < -0.1:
                st.error(f"Overall Market Sentiment: **Bearish** (Score: {avg_score:.2f})")
            else:
                st.info(f"Overall Market Sentiment: **Neutral** (Score: {avg_score:.2f})")
                
            st.write("#### Latest Headlines")
            for article in news_list:
                st.markdown(f"**[{article['title']}]({article['link']})** - *{article['publisher']}*")
                if article['score'] > 0:
                    st.write("🟢 Positive")
                elif article['score'] < 0:
                    st.write("🔴 Negative")
                else:
                    st.write("⚪ Neutral")
                st.markdown("---")
        else:
            st.warning("No recent news found for this ticker.")

with tab3:
    st.write("### 🔔 Price Alerts")
    st.write("Receive an email notification when your target price is triggered.")
    
    if "alerts_config" not in st.session_state:
        st.session_state["alerts_config"] = []
        
    with st.form("alert_form"):
        alert_email = st.text_input("Email Address", placeholder="you@example.com")
        alert_ticker = st.selectbox("Stock Ticker", tickers, index=tickers.index(selected_ticker) if selected_ticker in tickers else 0)
        col1, col2 = st.columns(2)
        alert_condition = col1.selectbox("Condition", ["Above", "Below"])
        
        default_target = realtime_price if realtime_price else 100.0
        alert_target = col2.number_input("Target Price ($)", min_value=0.01, value=float(default_target), step=1.0)
        
        submitted = st.form_submit_button("Set Alert")
        if submitted and alert_email:
            st.session_state["alerts_config"].append({
                "email": alert_email,
                "ticker": alert_ticker,
                "condition": alert_condition,
                "target": alert_target,
                "triggered": False
            })
            st.success(f"Alert set up successfully for {alert_ticker} {alert_condition} ${alert_target:.2f}!")
    
    if st.session_state["alerts_config"]:
        st.write("#### Active Alerts")
        for idx, alert in enumerate(st.session_state["alerts_config"]):
            status = "🔔 Triggered" if alert['triggered'] else "⏳ Pending"
            st.info(f"**{alert['ticker']}** {alert['condition']} ${alert['target']:.2f} -> {alert['email']} [{status}]")

with tab4:
    st.write("### 💼 AI Portfolio Suggester")
    st.write("Based on technical trend strength, we analyze popular tech & index stocks to provide a dynamically weighted portfolio.")
    
    if st.button("Generate Portfolio Suggestion"):
        with st.spinner("Analyzing technicals for multiple stocks..."):
            portfolio = generate_portfolio_suggestion(tickers)
            if portfolio:
                df_port = pd.DataFrame(portfolio)
                st.dataframe(df_port, use_container_width=True)
                
                # Pie chart
                fig_pie = go.Figure(data=[go.Pie(labels=df_port['Ticker'], values=df_port['Weight (%)'], hole=.3)])
                fig_pie.update_layout(title_text="Suggested Diversification", template="plotly_dark")
                st.plotly_chart(fig_pie)
            else:
                st.error("Not enough data to calculate portfolio trends.")
