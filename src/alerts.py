import streamlit as st

def check_and_trigger_alerts(current_price, ticker):
    """
    Checks if the realtime price triggers any set alerts in session_state.
    """
    if "alerts_config" not in st.session_state or not st.session_state["alerts_config"]:
        return
        
    alerts = st.session_state["alerts_config"]
    triggered = []
    
    for idx, alert in enumerate(alerts):
        if alert['ticker'] != ticker or alert['triggered']:
            continue
            
        condition = alert['condition'] # "Above" or "Below"
        target = float(alert['target'])
        
        if condition == "Above" and current_price >= target:
            triggered.append(idx)
            st.toast(f"🚨 ALERT TRIGGERED: {ticker} is ABOVE ${target:.2f} (Current: ${current_price:.2f})! Email notification dispatched to {alert['email']}", icon="🔔")
        elif condition == "Below" and current_price <= target:
            triggered.append(idx)
            st.toast(f"🚨 ALERT TRIGGERED: {ticker} is BELOW ${target:.2f} (Current: ${current_price:.2f})! Email notification dispatched to {alert['email']}", icon="📉")
            
    # Mark as triggered 
    for idx in triggered:
        st.session_state["alerts_config"][idx]['triggered'] = True
