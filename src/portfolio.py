import pandas as pd
from src.data_handler import fetch_data
from src.strategy import calculate_technical_indicators, determine_trend

def generate_portfolio_suggestion(tickers=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]):
    """
    Generates a mock portfolio suggestion based on current technical trends.
    """
    portfolio = []
    total_score = 0
    
    for ticker in tickers:
        df = fetch_data(ticker, period="6mo")
        if df is None or len(df) < 60:
            continue
            
        df_ind = calculate_technical_indicators(df)
        trend = determine_trend(df_ind)
        
        # Assign weight score based on trend
        score = 1.0 # default base weight
        if "Bullish" in trend:
            score = 2.5
        elif "Bearish" in trend:
            score = 0.5
            
        rsi = df_ind.iloc[-1]['RSI']
        # If oversold, might be a good buy opportunity
        if not pd.isna(rsi) and rsi < 30:
            score += 1.0
            
        total_score += score
        portfolio.append({'Ticker': ticker, 'Score': score, 'Trend': trend, 'Current Price': round(df_ind.iloc[-1]['Close'], 2)})
        
    if total_score == 0:
        return []
        
    # Normalize weights
    for item in portfolio:
        item['Weight (%)'] = round((item['Score'] / total_score) * 100, 1)
        
    # Sort by weight highest first
    portfolio = sorted(portfolio, key=lambda x: x['Weight (%)'], reverse=True)
    return portfolio
