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


def cal_slopes(log_returns):

    # Calculate the slope and R^2 for each cluster
    cluster_trends = {}
    for cluster, log_return in log_returns.items():
        # Generate time index for regression
        x = np.arange(len(log_return))
        
        # Fit linear regression and get the slope and R^2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_return)
        cluster_trends[cluster] = (slope, r_value**2)
        
    # Clusters with their corresponding slope and R^2
    trends_and_r_squared = {cluster: (slope, r_squared) for cluster, (slope, r_squared) in cluster_trends.items()}
    
    return trends_and_r_squared



def cal_vols(log_returns):

    # Define a function to fit GARCH model and calculate volatility
    def _cal_garch_volatility(data):
        model = arch_model(data, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp='off')
        volatility = model_fit.conditional_volatility
        return volatility

    # Calculate and store GARCH volatility for each cluster
    cluster_garch_volatility = {}
    for cluster, log_return in log_returns.items():
        # Rescale data
        scaled_log_return = 100 * log_return
        # Calculate the volatility with the rescaled data
        volatility = _cal_garch_volatility(scaled_log_return)
        cluster_garch_volatility[cluster] = volatility.iloc[-1] 

    vols = {cluster: vol for cluster, vol in cluster_garch_volatility.items()}

    return vols


def cal_mvs(log_returns, adjustment=None):

    # Calculate the Momentum Volatility Score (MVS) with normalized slopes and r_squared
    slopes_and_r_squared = cal_slopes(log_returns)
    risks = cal_vols(log_returns)
    momentum_volatility_score = {}
    slopes = {}

    # Check if adjustment is not None, then create sector_mvs_dict
    if adjustment is not None:
        sector_mvs_dict = {tag: mvs for tag, mvs in adjustment.items()}  
    else:
        sector_mvs_dict = {}

    for cluster in log_returns.keys():
        sector_tag = cluster.split()[0]   # extract sector tag
        slope, r_squared = slopes_and_r_squared[cluster]
        slopes[cluster] = slope

        # If adjustment is None then momentum_volatility_score[cluster] = slope * r_squared else multiply with sector_mvs_dict
        if adjustment is None:
            momentum_volatility_score[cluster] = slope * r_squared
        else:
            momentum_volatility_score[cluster] = slope * r_squared * sector_mvs_dict.get(sector_tag, 1)

    # Convert the momentum_volatility_score to a numpy array
    mvs_array = np.array(list(momentum_volatility_score.values())).reshape(-1, 1)

    # Create a scaler object and fit_transform with mvs_array
    scaler = MinMaxScaler()
    normalized_mvs = scaler.fit_transform(mvs_array)

    # Convert the normalized scores back to a dictionary
    normalized_mvs_dict = {cluster: score[0] for cluster, score in zip(momentum_volatility_score.keys(), normalized_mvs)}

    # Return a dictionary with all the values
    return {'mvs': normalized_mvs_dict, 'mom': slopes, 'vol': risks}


# Correlated Stocks Clustering and Ranking
def rank_stock_sectors(stocks, d, num_sectors=30, mvs_threshold=0.6):
    
    # Extract closing prices for each stock
    closing_prices = pd.DataFrame()
    data = []
    for stock in stocks.keys():
        dates = stocks[stock]['history'].keys()
        close_prices = [stocks[stock]['history'][date]['Close'] for date in dates]
        data.append(pd.Series(data=close_prices, index=dates, name=stock))
    closing_prices = pd.concat(data, axis=1)
    closing_prices.sort_index(inplace=True)
    log_returns = np.log(closing_prices/closing_prices.shift(1))
    log_returns = log_returns.iloc[1:]

    # Clustering
    corr_matrix = log_returns.tail(d).corr()
    Z = linkage(corr_matrix, 'ward')
    clusters = fcluster(Z, num_sectors, criterion='maxclust')
    corr_matrix['sector'] = clusters

    # Calculate the average log return of each sector
    group_avg_log_returns = {}
    for i in range(1, num_sectors+1):
        stocks = corr_matrix[corr_matrix['sector'] == i].index.tolist()
        group_data = log_returns[stocks].tail(d) # use tail here to get last 'd' days.
        avg_log_return = group_data.mean(axis=1)
        # add a 0 at the beginning of each series
        avg_log_return = pd.concat([pd.Series([0]), avg_log_return.cumsum()]).reset_index(drop=True)
        group_avg_log_returns[f"Sector {i}"] = avg_log_return[1:] 
        
    # Calculate Momentum, Volatility, and MVS of each seector
    mvs_result = cal_mvs(group_avg_log_returns)
    
    # Get the top & bottom clusters
    sorted_mvs = sorted(mvs_result['mvs'].items(), key=lambda x: x[1])
    bottom_clusters = sorted_mvs[:5]
    top_clusters = [clust for clust in sorted_mvs[-5:] if clust[1] >= mvs_threshold]
    bottom_clusters_names = [cluster for cluster, _ in bottom_clusters]
    top_clusters_names = [cluster for cluster, _ in top_clusters]
    all_clusters = group_avg_log_returns.keys()

    # Initialize target_sectors dictionary
    target_sectors = {
        'best': top_clusters[-1] if top_clusters else None,
        'top': top_clusters if top_clusters else None,
        'bottom': bottom_clusters if bottom_clusters else None,
    }
    
    # Visualisation
    fig = go.Figure()
    for cluster in all_clusters:
        if cluster == target_sectors['best'][0]:  # best cluster
            color = "#FBC546"
            width = 5
        elif cluster in top_clusters_names and cluster != target_sectors['best'][0]:  # top but not best
            color = "green"
            width = 4
        elif cluster in bottom_clusters_names:  # bottom clusters
            color = "red"
            width = 2
        else:  # other clusters
            color = "blue"
            width = 0.5
        fig.add_trace(go.Scatter(x=group_avg_log_returns[cluster].index, y=group_avg_log_returns[cluster],
                                mode='lines', name=cluster, line=dict(color=color, width=width), showlegend=False))

    fig.update_layout(
        autosize=True,
        title=f'Sectors Average Log Returns in the past {d} Days',
        title_x=0.5, 
        xaxis_title='Time',
        yaxis_title='Cumulative Average Log Returns',
        hovermode="x",
        template="plotly_white",
        margin=dict(
            l=10,  # left margin
            r=10,  # right margin
            pad=10  # padding
        )
    )
        
    return corr_matrix, group_avg_log_returns, mvs_result, target_sectors, fig



# Ranking Stock Individuals in sectors
def rank_stock_individuals(stocks, corr_matrix, target_sectors, d, mvs_threshold = 0, top_mvs_n=10):

    # Generate the dictionary of stocks within each sector
    target_sector_stocks = {sector[0]:corr_matrix[corr_matrix['sector'] ==int(sector[0].split()[-1])].index.tolist() for sector in target_sectors['top']}
  
    # Aggregate all stocks together
    all_stocks_ls = []
    for sector_name, stocks_ls in target_sector_stocks.items():
        all_stocks_ls.extend(stocks_ls)
    all_stocks = stocks[all_stocks_ls]

    # Extract closing prices for each stock
    closing_prices = pd.DataFrame()
    data = []
    for stock in all_stocks.keys():
        dates = all_stocks[stock]['history'].keys()
        close_prices = [all_stocks[stock]['history'][date]['Close'] for date in dates]
        data.append(pd.Series(data=close_prices, index=dates, name=stock))
        
    closing_prices = pd.concat(data, axis=1)
    closing_prices.sort_index(inplace=True)
    
    # Calculate log returns
    log_returns = np.log(closing_prices/closing_prices.shift(1))
    log_returns = log_returns.iloc[1:]
    log_returns = log_returns.tail(d)

    # Calculate the MVS for all stocks
    stock_adjustments = {}
    for sector, mvs in target_sectors['top']:
        stocks_in_sector = corr_matrix[corr_matrix['sector'] == int(sector.split()[-1])].index.tolist()
        for stock in stocks_in_sector:
            stock_adjustments[stock] = mvs 

    # Now, replace `target_sectors['top']` with `stock_adjustments` as the adjustment argument in your cal_mvs function
    all_stock_mvs_data = cal_mvs(log_returns, stock_adjustments)

    # Get stocks where mvs is higher than the threshold
    selected_stocks_list = [k for k, v in all_stock_mvs_data['mvs'].items() if v > mvs_threshold]
    selected_stocks_dict = {
        'mvs': {k: all_stock_mvs_data['mvs'][k] for k in selected_stocks_list},
        'mom': {k: all_stock_mvs_data['mom'][k] for k in selected_stocks_list},
        'vol': {k: all_stock_mvs_data['vol'][k] for k in selected_stocks_list}
    }      
        
    # Sort the selected stocks based on MVS in ascending order
    sorted_stocks_based_on_mvs =  sorted(selected_stocks_dict['mvs'].items(), key=lambda x: x[1])

    # Get the top stocks
    top_stocks = sorted_stocks_based_on_mvs[-top_mvs_n:]
    
    # Get the best stock
    best_stock = sorted_stocks_based_on_mvs[-1]

    target_stocks = {
        'best': best_stock,
        'top': top_stocks,
    }

    # Calculate the cumulative log returns
    cum_log_returns = log_returns.cumsum()

    # Visualization
    fig = go.Figure()
    for stock in [k for k, v in top_stocks]:
        if stock == target_stocks['best'][0]:  # Best stock
            color = "#FBC546"
            width = 5
        elif stock in [x[0] for x in target_stocks['top']]:  # Top but not best stocks
            color = "green"
            width = 4
        else:  # Other stocks
            color = "blue"
            width = 0.5
    
        fig.add_trace(go.Scatter(x=cum_log_returns.index, y=cum_log_returns[stock],
                                 mode='lines', name=stock, line=dict(color=color, width=width), showlegend=False))

    fig.update_layout(
        autosize=True,
        title=f'Top performance Stocks in the past {d} Days',
        title_x=0.5, 
        xaxis_title='Time',
        yaxis_title='Cumulative Log Returns',
        hovermode="x",
        template="plotly_white",
        margin=dict(
            l=10,  # Left margin
            r=10,  # Right margin
            pad=10  # Padding
        )
    )
    
    return corr_matrix, log_returns, all_stock_mvs_data, target_stocks, [k for k, v in top_stocks], stock_adjustments, fig

