import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
from scipy import stats
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import math
from plotly.subplots import make_subplots


def cal_ma(data, hf_d, st_d, mt_d, lt_d):
    # Calculate moving averages
    hf_ma = data['close'].rolling(window=hf_d).mean().iloc[hf_d-1:]
    st_ma = data['close'].rolling(window=st_d).mean().iloc[st_d-1:]
    mt_ma = data['close'].rolling(window=mt_d).mean().iloc[mt_d-1:]
    lt_ma = data['close'].rolling(window=lt_d).mean().iloc[lt_d-1:]

    # Get the last value in the series
    last_hf_ma = hf_ma.iloc[-1]
    last_st_ma = st_ma.iloc[-1]
    last_mt_ma = mt_ma.iloc[-1]
    last_lt_ma = lt_ma.iloc[-1]

    # Add indicators
    golden_lt = last_lt_ma < min(last_hf_ma, last_st_ma, last_mt_ma)
    golden_mt = last_mt_ma < min(last_hf_ma, last_st_ma)
    golden_st = last_st_ma < last_hf_ma

    # Create a dictionary for return
    ma_result_dict = {
        "hf_ma": hf_ma,
        "st_ma": st_ma,
        "mt_ma": mt_ma,
        "lt_ma": lt_ma,
        "last_hf_ma": last_hf_ma,
        "last_st_ma": last_st_ma,
        "last_mt_ma": last_mt_ma,
        "last_lt_ma": last_lt_ma,
        "golden_lt": golden_lt,
        "golden_mt": golden_mt,
        "golden_st": golden_st
    }

    return ma_result_dict



def cal_atr(data, hf_d):
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(hf_d).mean().iloc[hf_d-1:]

    atr_result_dict = {
        'last_hf_atr': atr.iloc[-1],
        'hf_atr': atr
        }

    return atr_result_dict



def plot_portfolio_individual(data, ma_result, atr_result, d):
    # Slice data and moving averages to keep only last 'd' days
    data = data.tail(d)

    # Create subplot with different height ratios
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.7, 0.15, 0.15]    # Adjust these values as needed
    )

    # Extract moving averages from results and slice for 'd' days
    hf_ma = ma_result['hf_ma'].tail(d)
    st_ma = ma_result['st_ma'].tail(d)
    mt_ma = ma_result['mt_ma'].tail(d)
    lt_ma = ma_result['lt_ma'].tail(d)
    hf_atr = atr_result['hf_atr'].tail(d)    # Extract atr values

    # Add traces
    fig.add_trace(go.Candlestick(x=data.index,
                                  open=data['open'],
                                  high=data['high'],
                                  low=data['low'],
                                  close=data['close'],
                                  name="OHLC Prices"), row=1, col=1)
                                  
    fig.add_trace(go.Scatter(x=hf_ma.index, y=hf_ma, mode='lines', name=f'HF MA', line={'color': 'blue'}), row=1, col=1)
    fig.add_trace(go.Scatter(x=st_ma.index, y=st_ma, mode='lines', name=f'ST MA', line={'color': 'green'}), row=1, col=1)
    fig.add_trace(go.Scatter(x=mt_ma.index, y=mt_ma, mode='lines', name=f'MT MA', line={'color': 'orange'}), row=1, col=1)
    fig.add_trace(go.Scatter(x=lt_ma.index, y=lt_ma, mode='lines', name=f'LT MA', line={'color': 'purple'}), row=1, col=1)

    # Add ATR to the subplot
    fig.add_trace(go.Scatter(x=hf_atr.index, y=hf_atr, mode='lines', name=f'ATR', line={'color': 'black'}), row=2, col=1)

    # Add Volume to the subplot
    fig.add_trace(go.Bar(x=data.index, y=data['vol'], showlegend=False, name="Volume"), row=3, col=1)

    # Formatting the plot
    fig.update_layout(
        autosize=True,
        title=f'Stock Price Series, Volume, and ATR in the past {d} days',
        title_x=0.5,
        hovermode="x",
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)

    return fig


def cal_mpt(stocks_data, selected_stocks_ls, mpt_d, hold_d):
    def confidence_interval(returns, volatility):
        CI_lower = returns - 1.96 * volatility
        CI_upper = returns + 1.96 * volatility
        return CI_lower, CI_upper

    # Initiate an empty DataFrame to store close prices data for selected stocks
    price_data = pd.DataFrame()

    # Loop through all selected stocks, process data and extract close prices
    for ticker in selected_stocks_ls:
        stock = stocks_data[ticker]['history']
        data = pd.DataFrame(stock).T
        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"
        data = data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high',
                                    'Low':'low', 'Volume':'vol'}).drop(columns=['Dividends', 'Stock Splits'])
        
        # get the last mpt_d close prices
        price_data[ticker] = data['close'].tail(mpt_d)

    # Calculate expected returns and the covariance matrix of the assets
    mu = expected_returns.mean_historical_return(price_data, frequency=252) # this annualizes the returns
    S = risk_models.sample_cov(price_data, frequency=252) # this annualizes the covariances

    # Set up the optimization problem to maximize Sharpe ratio
    ef = EfficientFrontier(mu, S)

    # Calculate the raw weights assuming zero risk-free rate
    raw_weights = ef.max_sharpe(risk_free_rate=0.0)

    # Clean raw weights, rounding small weights to zero and rounding others
    # to ensure the weights sum to exactly 1.0
    cleaned_weights = ef.clean_weights()

    # Calculate the portfolio performance
    annual_return, annual_volatility, annual_sharpe_ratio = ef.portfolio_performance(verbose=False, risk_free_rate=0.0)
    
    # Adjust performance to the hold_d holding period
    hold_return = annual_return / (252/hold_d)
    hold_volatility = annual_volatility / math.sqrt(252/hold_d)
    hold_performance = (hold_return, hold_volatility, annual_sharpe_ratio)

    # Calculate the 95% confidence interval
    CI_lower, CI_upper = confidence_interval(hold_return, hold_volatility)

    return cleaned_weights, hold_performance, (CI_lower, CI_upper)

