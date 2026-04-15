# 📈 AI-Based Sri Lankan & Global Stock Market Decision System

Welcome to the **AI-Based Stock Market Decision System**. This application does not just predict price, it acts as a decision engine processing historical trends, training Machine Learning algorithms in real-time, and giving definitive `BUY / SELL` recommendations based on a confluence of ML predictions and proven technical indicators (MA, RSI).

## 🚀 Key Features

1. **Intelligent Price Prediction**: Uses both baseline (Linear Regression) and deep learning (LSTM) models.
2. **Trend Detection**: Incorporates standard technical indicators (50-Day SMA, 200-Day SMA, and RSI) to gauge market sentiment.
3. **Actionable Signals**: Translates complex data into clear actions (`STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`).
4. **Confidence Scoring**: Provides a dynamic probability of prediction accuracy based on historical volatility and model fit.
5. **Interactive Dashboard**: Powered by **Streamlit**, boasting fluid, responsive charts with markers for AI predictions.

## ⚙️ Tech Stack
- **Language:** Python
- **Data & Math:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (Linear Regression), TensorFlow/Keras (LSTM)
- **Data Source:** Yahoo Finance (`yfinance`) - Defaults to robust global markets for highest accuracy.
- **Visuals:** Streamlit, Plotly
- **Technical Analysis:** `ta` library

## 🛠️ How to Build and Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Kajanthaa1/stock-market-predictor-ml.git
cd stock-market-predictor-ml
```

**2. Install dependencies**
We highly recommend using a Virtual Environment.
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit Application**
```bash
streamlit run app.py
```

## 🧠 Decision Logic (Why it works)
The unique selling point of this tool is its hybrid nature:
`IF predicted_price > current_price AND trend == bullish: BUY`

We factor in technical *momentum* alongside deep learning *forecasting*. It recognizes that a predicted upward tick during a deeply bearish macro-trend is a risky `HOLD` rather than a guaranteed `BUY`. 

## ⚖️ Disclaimer
*This project is built for educational and demonstration purposes. Do not use its output for real-world financial decision making without conducting your own extensive due diligence.*
