# Bitcoin Price forecasting
## Project Overview
This project investigates whether Bitcoin prices can be forecasted using historical market data. Using five years of Bitcoin price data, the study explores statistical and deep learning approaches for time series forecasting.

The project applies both classical time-series models (ARIMA) and deep learning models (LSTM) to predict Bitcoin log returns and compares their performance against a naive baseline model.

The analysis is accompanied by an interactive Streamlit dashboard that allows users to explore price trends, returns, and model forecasts.

## Data set
The dataset contains 5 years of Bitcoin historical price data (BTC-USD).

Source: Yahoo Finance

The dataset includes:

Date

Bitcoin closing price

From the closing price, log returns were calculated for time series modeling.

## Methodology
1. Data Collection

Bitcoin historical data was downloaded and cleaned for analysis.

2. Exploratory Data Analysis

The following analyses were performed:

Bitcoin price trend visualization

Rolling mean and rolling standard deviation

Log return transformation

Volatility inspection

3. Stationarity Testing

An Augmented Dickey–Fuller (ADF) test was applied.

Findings:

Bitcoin price series → non-stationary

Log return series → stationary

This confirmed that forecasting models should be applied to returns instead of raw prices.

4. Autocorrelation Analysis

ACF and PACF plots were used to examine temporal dependencies in the return series.

Result:
Autocorrelation in Bitcoin returns was found to be very weak, indicating limited predictive structure.

## Models that I used
1. ARIMA model - a classic statistical model for time-series forecasting
2. LSTM model - a deep learning model designed for sequential data
( This LSTM model was implemented to test whether non-linear neural networks could capture the patterns that cannot be detected by ARIMA model)

## Model Evaluation
Model were evaluated by using 
- RMSE (Root Mean Squared Error)
- MAE ( Mean Absolule Error)

## REsults
Baseline ≈ ARIMA ≈ LSTM

This indicates that historical returns alone provide very limited predictive power for short-term Bitcoin forecasting.

## Weekly Experiment
To reduce the noise in the daily data, the analysis was repeated using weekly returns.
Result: it also doesn't improve compared to the baseline.

## Key insights
- Bitcoin prices are non-stationary.
- log return are stationary.
- Autocorrelation in returns is very weak
- Complex models did not outperform a naive baseline
- Historical price data alone is insufficient for reliable short-term forecasting

These findings align with the Weak Form Efficient Market Hypothesis, which suggests that past price information cannot reliably predict future returns.

## Live Demo
https://05bitcoinforecasting-kztet3qbmdtuur4gffhw6p.streamlit.app/

## Libraries that used
- Python
- pandas
- Numpy
- Matplotlib
- Statsmodels
- Streamlit



