import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stasmodels.tsa.arima.model import ARIMA

st.title("Bitcoin Price Forecast Dashboard")

st.write("This dashboard shows Bitcoin price trends and ARIMA forecasting.")

# Load data
df = pd.read_csv("C:/Projects/05_bitcoin_forecasting/data/btc_cleaned.csv", header=[0,1], index_col=0, parse_dates=True)
df.columns = ["Close"]

# Price chart
st.subheader("Bitcoin Proce (5 Years)")

fig, ax = plt.subplots()
ax.plot(df.index, df["Close"])
ax.set_title("Bitcoin Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
st.pyplot(fig)

# Compute Log returns
returns = np.log(df["Close"]/df["Close"].shift(1))
returns = returns.dropna()

# Plot returns
st.subheader("Daily Log Returns")

fig2, ax2 = plt.subplots()
ax2.plot(returns)
ax2.set_title("Bitcoin Log Returns")
st.pyplot(fig2)

# Train ARIMA
data = returns.values
train_size = int(len(data) * 0.8)

train = data [:train_size]
test = data [train_size:]

model = ARIMA(train, order=(1,0,1))
fit = model.fit()

forecast = fit.forecast(steps=len(test))

# Plot forecast
st.subheader("ARIMA Forecast vs Actual")

fig3, ax3 = plt.subplots()
ax3.plot(test, label ="Actual")
ax3.plot(forecast, label="Forecast")
ax3.legend()
st.pyplot(fig3)

st.write("Forecast generated using ARIMA model.")