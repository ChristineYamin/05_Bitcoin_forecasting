import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Page Config
st.set_page_config(page_title="Bitcoin Forecast Dashboard", layout="wide")
st.title("Bitcoin Price Forecast Dashboard")
st.write("This dashboard shows Bitcoin price trends and ARIMA forecasting.")

# Load data
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = ["Close"]
    df = df.sort_index()
    return df

csv_path = "data/btc_cleaned.csv"
df = load_data(csv_path)


# Add a sidebar

st.sidebar.header("Settings")

years = st.sidebar.slider(
    "Select number of years of data",
    min_value=1,
    max_value=5,
    value=5
)

# Approx filter (crypto trades daily; 365 is fine for dashboard)
df_view = df.tail(years * 365).copy()


# Compute Log returns
returns = np.log(df_view["Close"] / df_view["Close"].shift(1)).dropna()

# Show Key Metrics
st.subheader("Key Statistics")
col1,col2, col3 = st.columns(3)
col1.metric("Latest Price", f"${df_view['Close'].iloc[-1]:,.2f}")
col2.metric("Mean Return", f"{returns.mean():.5f}")
col3.metric("Volatility", f"{returns.std():.5f}")

# Add a Data Table Viewer
with st.expander("Show raw data (last 20 rows)"):
    st.dataframe(df_view.tail(20))



# Price chart
st.subheader(f"Bitcoin Price (Last {years} Years)")

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_view.index, df_view["Close"])
ax.set_title("Bitcoin Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(fig, use_container_width=False)


# Returns chart
st.subheader("Daily Log Returns")

fig2, ax2 = plt.subplots(figsize=(5,3))
ax2.plot(returns.index, returns.values)
ax2.set_title("Bitcoin Log Returns")
ax2.set_xlabel("Date")
ax2.set_ylabel("Log return")
ax2.legend()
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(fig2, use_container_width=False)

# ARIMA modeling

st.subheader("ARIMA Forecast vs Actual (Log Returns)")

data = returns.values
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]

# Use matching index for plotting
test_index = returns.index[train_size:]

model = ARIMA(train, order=(1, 0, 1))
fit = model.fit()

forecast = fit.forecast(steps=len(test))

fig3, ax3 = plt.subplots(figsize=(5,3))
ax3.plot(test_index, test, label="Actual", linewidth=1, alpha=0.7)
ax3.plot(test_index, forecast, label="Forecast", linewidth=2)
ax3.set_title("ARIMA Forecast vs Actual")
ax3.set_xlabel("Date")
ax3.set_ylabel("Log Return")
ax3.legend()
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.pyplot(fig3, use_container_width=False)

st.caption("Note: Bitcoin returns are often close to random noise, so forecasts may stay near zero.")