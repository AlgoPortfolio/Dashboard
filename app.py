# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

# Streamlit app title
st.title("Trade Like a Quant Portfolio App")

# --- Data Pull Function ---
@st.cache_data
def fetch_yahoo_data(tickers, start_date="2010-01-01", end_date=str(datetime.today().date())):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].reset_index()
            df['ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Close': 'close', 'Adj Close': 'unadjusted_close', 'Date': 'date'})
            data[ticker] = df[['date', 'ticker', 'close', 'unadjusted_close', 'Volume']]
        except Exception as e:
            st.warning(f"Failed to fetch {ticker}: {e}")
    prices = pd.concat(data.values(), ignore_index=True)
    prices = prices.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    # VIX data
    vix_data = data.get('^VIX', pd.DataFrame()).rename(columns={'close': 'VIX'})
    vix3m_data = data.get('^VIX3M', pd.DataFrame()).rename(columns={'close': 'VIX3M'})
    if not vix_data.empty and not vix3m_data.empty:
        vol_idxs = vix_data[['date', 'VIX']].merge(vix3m_data[['date', 'VIX3M']], on='date', how='inner')
        vol_idxs['Basis'] = vol_idxs['VIX3M'] - vol_idxs['VIX']
        vol_idxs['ivts'] = vol_idxs['VIX'] / vol_idxs['VIX3M']
        vol_idxs['vixvol'] = vol_idxs['VIX'].rolling(window=60, min_periods=60).std() * np.sqrt(252)
        vol_idxs['vix3mvol'] = vol_idxs['VIX3M'].rolling(window=60, min_periods=60).std() * np.sqrt(252)
    else:
        vol_idxs = pd.DataFrame()
    
    return prices, vol_idxs

# --- Volatility Calculation ---
def calculate_volatility(returns, lookback):
    return returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)

def calculate_strategy_volatility(positions, returns, lookback):
    strat_returns = (positions.shift(1) * returns).sum(axis=1)
    return strat_returns.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)

# --- Strategy Functions ---
def risk_premia_strategy(prices, target_vols, lookback):
    df = prices.copy()
    df['returns'] = df.groupby('ticker')['close'].pct_change().apply(lambda x: np.log(1 + x))
    df['vol'] = df.groupby('ticker')['returns'].apply(lambda x: calculate_volatility(x, lookback))
    
    theosize = pd.DataFrame(index=df.index)
    for ticker in target_vols.keys():
        ticker_mask = df['ticker'] == ticker
        theosize.loc[ticker_mask, ticker] = target_vols[ticker] / df.loc[ticker_mask, 'vol']
        theosize.loc[ticker_mask, ticker] = theosize.loc[ticker_mask, ticker].fillna(0)
    
    df = df.merge(theosize, left_index=True, right_index=True)
    return df

def vrp_strategy(prices, vol_idxs, target_vol, lookback):
    df = prices[prices['ticker'].isin(['SVXY', 'VIXY'])].copy()
    df = df.merge(vol_idxs[['date', 'VIX3M', 'vixvol', 'ivts']], on='date', how='left')
    
    df['short_hurdle'] = df['VIX3M'].apply(lambda x: 0.85 if x < 15 else 0.9 if x < 17 else 0.95 if x < 20 else 1.0 if x < 25 else 1.1)
    df['long_hurdle'] = df.apply(lambda x: x['short_hurdle'] if x['vixvol'] <= 20 else 1.1, axis=1)
    
    df['position'] = 0
    df.loc[df['ivts'] <= df['short_hurdle'], 'position'] = -1  # Short SVXY
    df.loc[df['ivts'] >= df['long_hurdle'], 'position'] = 1   # Long VIXY
    df.loc[df['ticker'] == 'SVXY', 'position'] = df.loc[df['ticker'] == 'SVXY', 'position'].where(df['position'] == -1, 0)
    df.loc[df['ticker'] == 'VIXY', 'position'] = df.loc[df['ticker'] == 'VIXY', 'position'].where(df['position'] == 1, 0)
    
    # Adjust SVXY post-2018-02-27
    df.loc[(df['date'] > '2018-02-27') & (df['ticker'] == 'SVXY'), 'position'] *= 2
    
    df['returns'] = df.groupby('ticker')['close'].pct_change().apply(lambda x: np.log(1 + x))
    positions = df.pivot(index='date', columns='ticker', values='position').fillna(0)
    returns = df.pivot(index='date', columns='ticker', values='returns').fillna(0)
    strat_vol = calculate_strategy_volatility(positions, returns, lookback)
    
    df = df.merge(strat_vol.rename('strat_vol'), on='date')
    df['theosize'] = target_vol / df['strat_vol']
    df['theosize'] = df['theosize'].fillna(0)
    
    return df

# --- Backtest Function ---
def run_backtest(prices, rp_df, vrp_df, portfolio_value, max_leverage):
    combined = pd.concat([
        rp_df[rp_df['ticker'].isin(['VTI', 'TLT', 'GLD'])][['date', 'ticker', 'VTI', 'TLT', 'GLD', 'close']].rename(columns={'VTI': 'rp_VTI', 'TLT': 'rp_TLT', 'GLD': 'rp_GLD'}),
        vrp_df[vrp_df['ticker'].isin(['SVXY', 'VIXY'])][['date', 'ticker', 'theosize', 'position', 'close']]
    ])
    
    combined = combined.pivot_table(
        index='date', 
        columns='ticker', 
        values=['rp_VTI', 'rp_TLT', 'rp_GLD', 'theosize', 'position', 'close'],
        aggfunc='first'
    ).reset_index()
    combined.columns = ['_'.join(col).strip() if col[1] else col[0] for col in combined.columns]
    
    for ticker in ['VTI', 'TLT', 'GLD']:
        combined[f'theosize_{ticker}'] = combined[f'rp_{ticker}'].fillna(0)
    for ticker in ['SVXY', 'VIXY']:
        combined[f'theosize_{ticker}'] = combined[f'theosize_{ticker}'] * combined[f'position_{ticker}']
    
    total_size = combined[[f'theosize_{t}' for t in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']]].abs().sum(axis=1)
    combined['adj_factor'] = np.where(total_size > max_leverage, max_leverage / total_size, 1)
    
    portfolio = pd.DataFrame(index=combined['date'])
    portfolio['value'] = portfolio_value
    portfolio['returns'] = 0.0
    
    for ticker in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']:
        size = combined[f'theosize_{ticker}'] * combined['adj_factor']
        price = combined[f'close_{ticker}']
        shares = (portfolio_value * size / price).round()
        returns = price.pct_change().apply(lambda x: np.log(1 + x)).fillna(0)
        portfolio['returns'] += (shares.shift(1) * price * returns / portfolio['value']).fillna(0)
    
    portfolio['value'] = portfolio_value * (1 + portfolio['returns']).cumprod()
    portfolio['drawdown'] = (portfolio['value'] / portfolio['value'].cummax()) - 1
    
    return portfolio, combined

# --- Input Controls ---
st.sidebar.header("Portfolio Settings")
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=1000)
max_leverage = st.sidebar.slider("Max Leverage", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

st.sidebar.header("Risk Premia (RP) Parameters")
rp_vol_stock = st.sidebar.slider("Stock (VTI) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_vol_bond = st.sidebar.slider("Bond (TLT) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_vol_gold = st.sidebar.slider("Gold (GLD) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_lookback = st.sidebar.slider("RP Volatility Lookback (days)", 10, 100, 30, 5)

st.sidebar.header("Volatility Risk Premia (VRP) Parameters")
vrp_vol = st.sidebar.slider("VRP Target Volatility (%)", 0.0, 10.0, 2.0, 0.1) / 100
vrp_lookback = st.sidebar.slider("VRP Volatility Lookback (days)", 60, 500, 120, 10)

# --- Fetch Data ---
tickers = ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY', '^VIX', '^VIX3M']
prices, vol_idxs = fetch_yahoo_data(tickers)

if prices.empty or vol_idxs.empty:
    st.error("Failed to load data. Please try again.")
else:
    # --- Run Strategies ---
    target_vols = {'VTI': rp_vol_stock, 'TLT': rp_vol_bond, 'GLD': rp_vol_gold}
    rp_df = risk_premia_strategy(prices, target_vols, rp_lookback)
    vrp_df = vrp_strategy(prices, vol_idxs, vrp_vol, vrp_lookback)
    
    # --- Run Backtest ---
    portfolio, positions = run_backtest(prices, rp_df, vrp_df, portfolio_value, max_leverage)
    
    # --- Display Backtest ---
    st.header("Portfolio Performance")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['value'], mode='lines', name='Portfolio Value'))
    st.plotly_chart(fig)
    
    st.header("Performance Metrics")
    annualized_return = ((portfolio['value'].iloc[-1] / portfolio_value) ** (252 / len(portfolio))) - 1
    max_drawdown = portfolio['drawdown'].min()
    st.write(f"Annualized Return: {annualized_return:.2%}")
    st.write(f"Max Drawdown: {max_drawdown:.2%}")
    
    st.header("Position Sizes")
    pos_fig = px.line(positions, x='date', y=[f'theosize_{t}' for t in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']],
                      title="Position Sizes (After Leverage Constraint)")
    st.plotly_chart(pos_fig)
    
    st.header("Portfolio Data")
    st.dataframe(portfolio.tail())

