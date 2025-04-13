# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin
import math
import time

# Streamlit app title
st.title("Trade Like a Quant Portfolio App")

# --- Data Pull Function (Updated) ---
@st.cache_data
def fetch_yahoo_data(tickers, start_date="2010-01-01", end_date=str(datetime.today().date())):
    data = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    st.warning(f"No data for {ticker}")
                    continue
                required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing columns {missing_cols} for {ticker}. Using 'Close' as fallback.")
                    for col in missing_cols:
                        df[col] = df['Close'] if col == 'Adj Close' else 0
                df = df[required_cols].reset_index()
                df['ticker'] = ticker
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.rename(columns={'Close': 'close', 'Adj Close': 'unadjusted_close', 'Date': 'date'})
                data[ticker] = df[['date', 'ticker', 'close', 'unadjusted_close', 'Volume']]
                break
            except Exception as e:
                st.warning(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt == 2:
                    st.error(f"Failed to fetch {ticker} after 3 attempts.")
                time.sleep(1)
    if not data:
        st.error("No data fetched for any ticker. Check logs or try again later.")
        return pd.DataFrame(), pd.DataFrame()
    
    prices = pd.concat(data.values(), ignore_index=True)
    prices = prices.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    vix_data = data.get('^VIX', pd.DataFrame()).rename(columns={'close': 'VIX'})
    vix3m_data = data.get('^VIX3M', pd.DataFrame()).rename(columns={'close': 'VIX3M'})
    if not vix_data.empty:
        if not vix3m_data.empty:
            vol_idxs = vix_data[['date', 'VIX']].merge(vix3m_data[['date', 'VIX3M']], on='date', how='inner')
        else:
            st.warning("VIX3M data missing. Using VIX * 1.1 as proxy.")
            vol_idxs = vix_data[['date', 'VIX']].copy()
            vol_idxs['VIX3M'] = vol_idxs['VIX'] * 1.1
        vol_idxs['Basis'] = vol_idxs['VIX3M'] - vol_idxs['VIX']
        vol_idxs['ivts'] = vol_idxs['VIX'] / vol_idxs['VIX3M']
        vol_idxs['vixvol'] = vol_idxs['VIX'].rolling(window=60, min_periods=60).std() * np.sqrt(252)
        vol_idxs['vix3mvol'] = vol_idxs['VIX3M'].rolling(window=60, min_periods=60).std() * np.sqrt(252)
    else:
        st.error("VIX data missing. VRP strategy will be disabled.")
        vol_idxs = pd.DataFrame(columns=['date', 'VIX', 'VIX3M', 'Basis', 'ivts', 'vixvol', 'vix3mvol'])
    
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
    df['position'] = 1  # Long-only for RP
    return df

def flow_effects_strategy(prices, target_vol, lookback, trade_eom, trade_som, eom_day, som_hold_days, go_long_day, cover_short_day, stock_eom_day, long_only):
    df = prices[prices['ticker'].isin(['VTI', 'TLT'])].copy()
    df['returns'] = df.groupby('ticker')['close'].pct_change().apply(lambda x: np.log(1 + x))
    
    # Calculate trading days
    df['month'] = df['date'].dt.to_period('M')
    df['trading_day'] = df.groupby('month').cumcount() + 1
    df['max_trading_day'] = df.groupby('month')['trading_day'].transform('max')
    df['tdm'] = df['trading_day'] - df['max_trading_day']
    df['is_eom'] = (df['tdm'] == 0) & (df['trading_day'] >= eom_day)
    df['is_som'] = (df['trading_day'] == 1)
    df['som_hold_end'] = df['is_som'].shift(som_hold_days, fill_value=False)
    df['go_long_day'] = (df['trading_day'] >= go_long_day)
    df['cover_short_day'] = (df['trading_day'] >= cover_short_day)
    df['stock_eom_day'] = (df['trading_day'] >= stock_eom_day)
    
    # Signals
    df['position'] = 0
    # Window dressing: Short TLT on EOM
    if trade_eom:
        df.loc[(df['ticker'] == 'TLT') & (df['is_eom']), 'position'] = -1
        df.loc[(df['ticker'] == 'TLT') & (df['cover_short_day']), 'position'] = 0  # Cover short
    # SOM: Long VTI
    if trade_som:
        df.loc[(df['ticker'] == 'VTI') & (df['is_som']), 'position'] = 1
        df.loc[(df['ticker'] == 'VTI') & (df['som_hold_end']), 'position'] = 0  # End SOM hold
    # Mid-month rebalancing
    df['cum_returns'] = df.groupby('ticker')['returns'].cumsum()
    df = df.merge(
        df[df['trading_day'] == go_long_day][['date', 'ticker', 'cum_returns']],
        on=['date', 'ticker'], 
        how='left', 
        suffixes=('', '_go_long')
    )
    df['cum_returns_go_long'] = df.groupby('ticker')['cum_returns_go_long'].fillna(method='ffill')
    df_vti = df[df['ticker'] == 'VTI'][['date', 'cum_returns_go_long']].rename(columns={'cum_returns_go_long': 'vti_ret'})
    df_tlt = df[df['ticker'] == 'TLT'][['date', 'cum_returns_go_long']].rename(columns={'cum_returns_go_long': 'tlt_ret'})
    df = df.merge(df_vti, on='date', how='left').merge(df_tlt, on='date', how='left')
    df['outperform'] = df['tlt_ret'] > df['vti_ret']
    df.loc[(df['go_long_day']) & (df['outperform']) & (df['ticker'] == 'VTI'), 'position'] = 1
    # Stock EOM
    df.loc[(df['ticker'] == 'VTI') & (df['stock_eom_day']), 'position'] = 1
    
    if long_only:
        df['position'] = df['position'].clip(lower=0)
    
    # Volatility targeting
    positions = df.pivot(index='date', columns='ticker', values='position').fillna(0)
    returns = df.pivot(index='date', columns='ticker', values='returns').fillna(0)
    strat_vol = calculate_strategy_volatility(positions, returns, lookback)
    
    df = df.merge(strat_vol.rename('strat_vol'), on='date')
    df['theosize'] = target_vol / df['strat_vol']
    df['theosize'] = df['theosize'].fillna(0)
    
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
def run_backtest(prices, rp_df, flow_df, vrp_df, portfolio_value, max_leverage, long_only, rebalance_threshold, current_holdings):
    combined = pd.concat([
        rp_df[rp_df['ticker'].isin(['VTI', 'TLT', 'GLD'])][['date', 'ticker', 'VTI', 'TLT', 'GLD', 'position', 'close']],
        flow_df[flow_df['ticker'].isin(['VTI', 'TLT'])][['date', 'ticker', 'theosize', 'position', 'close']].rename(columns={'theosize': 'flow_size'}),
        vrp_df[vrp_df['ticker'].isin(['SVXY', 'VIXY'])][['date', 'ticker', 'theosize', 'position', 'close']].rename(columns={'theosize': 'vrp_size'})
    ])
    
    combined = combined.pivot_table(
        index='date', 
        columns='ticker', 
        values=['VTI', 'TLT', 'GLD', 'flow_size', 'vrp_size', 'position', 'close'],
        aggfunc='first'
    ).reset_index()
    combined.columns = ['_'.join(col).strip() if col[1] else col[0] for col in combined.columns]
    
    for ticker in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']:
        combined[f'theosize_{ticker}'] = 0
        if ticker in ['VTI', 'TLT', 'GLD']:
            combined[f'theosize_{ticker}'] += combined[f'VTI_{ticker}'].fillna(0)
            combined[f'theosize_{ticker}'] += combined[f'flow_size_{ticker}'].fillna(0) if ticker in ['VTI', 'TLT'] else 0
        if ticker in ['SVXY', 'VIXY']:
            combined[f'theosize_{ticker}'] += combined[f'vrp_size_{ticker}'].fillna(0)
        combined[f'position_{ticker}'] = combined[f'position_{ticker}'].fillna(0)
    
    if long_only:
        for ticker in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']:
            combined[f'theosize_{ticker}'] = combined[f'theosize_{ticker}'].clip(lower=0)
    
    total_size = combined[[f'theosize_{t}' for t in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']]].abs().sum(axis=1)
    combined['adj_factor'] = np.where(total_size > max_leverage, max_leverage / total_size, 1)
    
    # Calculate target units and deltas
    portfolio = pd.DataFrame(index=combined['date'])
    portfolio['value'] = portfolio_value
    portfolio['returns'] = 0.0
    deltas = {}
    
    for ticker in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']:
        size = combined[f'theosize_{ticker}'] * combined['adj_factor']
        price = combined[f'close_{ticker}']
        target_units = (portfolio_value * size / price).round().fillna(0)
        current_units = current_holdings.get(ticker, 0)
        delta = target_units - current_units
        deltas[ticker] = delta
        
        shares = target_units  # Use target units for backtest
        returns = price.pct_change().apply(lambda x: np.log(1 + x)).fillna(0)
        portfolio['returns'] += (shares.shift(1) * price * returns / portfolio['value']).fillna(0)
    
    # Rebalance logic
    rebalance = total_size > (rebalance_threshold / 100) * max_leverage
    combined['rebalance'] = rebalance
    
    portfolio['value'] = portfolio_value * (1 + portfolio['returns']).cumprod()
    portfolio['drawdown'] = (portfolio['value'] / portfolio['value'].cummax()) - 1
    
    return portfolio, combined, deltas

# --- Input Controls (Updated) ---
st.sidebar.header("Portfolio Settings")
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=1000)
max_leverage = st.sidebar.slider("Max Leverage", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
long_only_portfolio = st.sidebar.checkbox("Long Only Portfolio", value=True)
rebalance_threshold = st.sidebar.slider("Rebalance at % of Target", min_value=0, max_value=100, value=20, step=5)

# Current Holdings
st.sidebar.header("Current Holdings")
current_stock = st.sidebar.number_input("Current Stock ETF Units (VTI)", min_value=0, value=0, step=1)
current_bond = st.sidebar.number_input("Current Bond ETF Units (TLT)", min_value=0, value=0, step=1)
current_gold = st.sidebar.number_input("Current Gold ETF Units (GLD)", min_value=0, value=0, step=1)
current_long_vol = st.sidebar.number_input("Current Long Vol ETF Units (VIXY)", min_value=0, value=0, step=1)
current_short_vol = st.sidebar.number_input("Current Short Vol ETF Units (SVXY)", min_value=0, value=0, step=1)
current_holdings = {
    'VTI': current_stock,
    'TLT': current_bond,
    'GLD': current_gold,
    'VIXY': current_long_vol,
    'SVXY': current_short_vol
}

st.sidebar.header("Risk Premia (RP) Parameters")
rp_vol_stock = st.sidebar.slider("Stock (VTI) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_vol_bond = st.sidebar.slider("Bond (TLT) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_vol_gold = st.sidebar.slider("Gold (GLD) Target Volatility (%)", 0.0, 10.0, 3.5, 0.1) / 100
rp_lookback = st.sidebar.slider("RP Volatility Lookback (days)", 10, 100, 30, 5)

st.sidebar.header("Flow Effects Parameters")
flow_vol = st.sidebar.slider("Flow Target Volatility (%)", 0.0, 10.0, 3.0, 0.1) / 100
flow_lookback = st.sidebar.slider("Flow Volatility Lookback (days)", 10, 100, 30, 5)
trade_eom = st.sidebar.checkbox("Trade End of Month (EOM)", value=True)
eom_day = st.sidebar.slider("EOM Trading Day", min_value=1, max_value=21, value=15, step=1)
trade_som = st.sidebar.checkbox("Trade Start of Month (SOM)", value=True)
som_hold_days = st.sidebar.slider("SOM Hold Days", min_value=1, max_value=21, value=5, step=1)
go_long_day = st.sidebar.slider("Go Long on Trading Day", min_value=1, max_value=21, value=15, step=1)
cover_short_day = st.sidebar.slider("Cover Short on Trading Day", min_value=1, max_value=21, value=5, step=1)
stock_eom_day = st.sidebar.slider("Stock EOM Trading Day", min_value=1, max_value=21, value=15, step=1)
long_only_flow = st.sidebar.checkbox("Long Only Flow", value=True)

st.sidebar.header("Volatility Risk Premia (VRP) Parameters")
vrp_vol = st.sidebar.slider("VRP Target Volatility (%)", 0.0, 10.0, 2.0, 0.1) / 100
vrp_lookback = st.sidebar.slider("VRP Volatility Lookback (days)", 60, 500, 120, 10)

# --- Fetch Data ---
tickers = ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY', '^VIX', '^VIX3M']
prices, vol_idxs = fetch_yahoo_data(tickers)

if prices.empty or (vol_idxs.empty and not prices[prices['ticker'].isin(['SVXY', 'VIXY'])].empty):
    st.error("Failed to load data. Please try again or check logs.")
else:
    # --- Run Strategies ---
    target_vols = {'VTI': rp_vol_stock, 'TLT': rp_vol_bond, 'GLD': rp_vol_gold}
    rp_df = risk_premia_strategy(prices, target_vols, rp_lookback)
    flow_df = flow_effects_strategy(prices, flow_vol, flow_lookback, trade_eom, trade_som, eom_day, som_hold_days, go_long_day, cover_short_day, stock_eom_day, long_only_flow)
    vrp_df = vrp_strategy(prices, vol_idxs, vrp_vol, vrp_lookback)
    
    # --- Run Backtest ---
    portfolio, positions, deltas = run_backtest(prices, rp_df, flow_df, vrp_df, portfolio_value, max_leverage, long_only_portfolio, rebalance_threshold, current_holdings)
    
    # --- Display Backtest ---
    st.header("Portfolio Performance")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['value'], mode='lines', name='Portfolio Value'))
    fig.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig)
    
    st.header("Performance Metrics")
    days = (portfolio.index[-1] - portfolio.index[0]).days if len(portfolio) > 1 else 1
    annualized_return = ((portfolio['value'].iloc[-1] / portfolio_value) ** (252 / days)) - 1
    max_drawdown = portfolio['drawdown'].min()
    st.write(f"Annualized Return: {annualized_return:.2%}")
    st.write(f"Max Drawdown: {max_drawdown:.2%}")
    
    st.header("Position Sizes")
    pos_fig = px.line(positions, x='date', y=[f'theosize_{t}' for t in ['VTI', 'TLT', 'GLD', 'SVXY', 'VIXY']],
                      title="Position Sizes (After Leverage Constraint)")
    st.plotly_chart(pos_fig)
    
    st.header("Deltas (Target - Current Units)")
    for ticker, delta in deltas.items():
        st.write(f"{ticker}: {delta.iloc[-1] if not delta.empty else 0:.0f} units")
    
    st.header("Portfolio Data")
    st.dataframe(portfolio.tail())
