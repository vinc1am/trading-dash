from flask import Flask, render_template, jsonify, Response, request
from queue import Queue
import json
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
import pickle
import plotly.io as pio
from scripts.stocks_data_extraction import *
from scripts.mvs_model import *
from scripts.portfolio import *

# ==================================== #
#               Main
# ==================================== #
app = Flask(__name__, static_folder='assets')

logs_queue = Queue()
def add_sys_log(character, log_message, drop_last=False, color_index=0):
    log = {'character': character, 'message': log_message, 'drop_last':drop_last, 'color_index':color_index}
    logs_queue.put(json.dumps(log))

@app.route('/')
def home():
    return render_template('stocks.html')

@app.route('/add_log', methods=['POST'])
def add_log_endpoint():
    request_data = request.get_json()
    log_character = request_data.get('character', 'Anonymous')
    log_message = request_data.get('message', '')
    drop_last = request_data.get('drop_last', False)
    color_index = request_data.get('color_index', 0)
    add_sys_log(log_character, log_message, drop_last, color_index)
    return {"status": "log added successfully"}

@app.route('/stream_logs')
def stream_logs():
    def generate_logs():
        while True:
            log_message = logs_queue.get()  
            yield f"data:{log_message}\n\n"
            time.sleep(0.1)
    return Response(generate_logs(), mimetype='text/event-stream')


config = {}
def initialize_app():
    global config
    try:
        with open('data/config.json') as config_file:
            config = json.load(config_file)
        print("Config loaded successfully!")
    except IOError as e:
        print(f"Failed to load configuration: {e}")

    # Get current date
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d (%A)")

    # Add welcome log
    add_sys_log('', f'Welcome back! Today is {date_string}.')

initialize_app()




# ==================================== #
#        Stock Data Extraction
# ==================================== #
# Stock Data Extraction Page
@app.route('/stocks')
def stocks():
    return render_template('stocks.html')


# Extract SNP Stocks Data
@app.route('/load_stocks_data')
def load_stocks():
    
    # Stock List
    snp_stock_ls = get_snp500_stock_list()
    snp_stock_ls_json = snp_stock_ls.to_json(orient='records')
    with open('data/stocks_list.json', 'w') as f:
        f.write(snp_stock_ls_json)
    add_sys_log('', 'S&P Stocks List is updated.')

    # Stock Data
    today = datetime.now(pytz.timezone('UTC')).strftime('%Y%m%d')
    past = (datetime.now(pytz.timezone('UTC')) - timedelta(days=config['stock_extract_yrs']*365)).strftime('%Y%m%d') 
    start, end = past, today
    add_sys_log('', f'Extracting S&P Stocks Data from {start} to {end}.')

    stocks_data = {}
    total = len(snp_stock_ls['Symbol'])
    add_sys_log('', 'Extracting S&P Stocks Data.')
    # Stock Data
    for i, ticker in enumerate(snp_stock_ls['Symbol'], start=1):
        stock_info = get_stock_info(ticker, start, end)

        # Check if the 'history' key exists in stock_info and if the stock price history is larger than lt_d
        if 'history' not in stock_info or len(stock_info['history']) < config['lt_d']:
            print(f"Skipping {ticker} as it has less than {config['lt_d']} days of data.")
            continue

        stocks_data[ticker] = stock_info

        if i == 0:
            add_sys_log('', '')
            add_sys_log('', f'[{i}/{total}] {ticker}')
        else:
            add_sys_log('', f'[{i}/{total}] {ticker}', True)

    with open('data/stocks_data.json', 'w') as f:
        json.dump(stocks_data, f)
    add_sys_log('', 'All S&P Stocks Data are extracted.')
    return {"status": "Extracted the SNP Stock Data"}


# Get the SNP Stocks Data
@app.route('/get_stocks_data')
def get_stocks():

    with open('data/stocks_list.json', 'r') as f:
        stocks_json = f.read()
    stocks = pd.read_json(stocks_json, orient='records')

    with open('data/stocks_data.json', 'r') as f:
        stocks_data_json = f.read()
    stocks_data = pd.read_json(stocks_data_json, orient='records')

    # Calculate sector distribution
    sector_distribution = stocks['GICS Sector'].value_counts().to_dict()
    add_sys_log('', 'Loading the S&P Stocks Data.')
    
    # Return sector distribution, stocks list and stocks data
    return {
        'stocks': stocks.to_dict('records'),
        'sectorDistribution': sector_distribution,
        'stocksData': stocks_data.to_dict('records'),
        'st_d': config['st_d'],
        'lt_d': config['lt_d'],
    }
    
    
    
# ==================================== #
#            Sectors MVS 
# ==================================== #    
# Top MVS Sectors Page
@app.route('/top_sectors')
def top_sectors():
    return render_template('top_sectors.html')

# Calculate Sectors MVS
@app.route('/cal_sectors_mvs')
def cal_sectors_mvs():

    add_sys_log('', 'Calculating the Sectors MVS.')
    with open('data/stocks_list.json', 'r') as f:
        stocks_json = f.read()
    stocks = pd.read_json(stocks_json, orient='records')

    with open('data/stocks_data.json', 'r') as f:
        stocks_data_json = f.read()
    stocks_data = pd.read_json(stocks_data_json, orient='records')

    corr_matrix, group_avg_log_returns, mvs_result, target_sectors, fig = rank_stock_sectors(stocks_data, 
                                                                                             num_sectors=config['num_sectors'], 
                                                                                             d=config['lt_d'], 
                                                                                             mvs_threshold=config['mvs_threshold'])

    all_sector_stocks = {f'Sector {i}':corr_matrix[corr_matrix['sector'] == i].index.tolist() for i in corr_matrix['sector'].unique()}
    
    all_sector_stocks_gics = {}
    for sector in all_sector_stocks:
        all_sector_stocks_gics[sector] = stocks.loc[
            stocks['Symbol'].isin(all_sector_stocks[sector]), 
            'GICS Sector'
        ].value_counts().to_dict()
    
    save_data = {
        'stocks': stocks.to_dict('records'),
        'corr_matrix': corr_matrix.to_json(orient='split'),
        'mvs_result': mvs_result,
        'target_sectors': target_sectors,
        'ai_sectors': all_sector_stocks,
        'ai_sectors_gics': all_sector_stocks_gics,
        'ai_sectors_fig': fig.to_json() 
    }

    # Save your desired data into a pickle file
    with open('data/top_sectors_mvs_result.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    add_sys_log('', 'Calculated MVS and saved results.')
    return save_data


@app.route('/load_sector_mvs')
def load_sector_mvs():
    with open('data/top_sectors_mvs_result.pkl', 'rb') as f:
        top_sectors_mvs_result = pickle.load(f)
    return jsonify(top_sectors_mvs_result)



# ==================================== #
#            Stocks MVS 
# ==================================== #      
# Top MVS Stocks Page
@app.route('/top_stocks')
def top_stocks():
    return render_template('top_stocks.html')

# Calculate Stocks MVS
@app.route('/cal_stocks_mvs')
def cal_stocks_mvs():

    add_sys_log('', 'Retrieving the Sectors MVS result.')
    with open('data/top_sectors_mvs_result.pkl', 'rb') as f:
        top_sectors_mvs_result = pickle.load(f)
    top_sectors_mvs_result['corr_matrix'] = pd.read_json(top_sectors_mvs_result['corr_matrix'], orient='split')

    add_sys_log('', 'Calculating the Stocks MVS.')
    with open('data/stocks_data.json', 'r') as f:
        stocks_data_json = f.read()
    stocks_data = pd.read_json(stocks_data_json, orient='records')

    corr_matrix, log_returns, all_stock_mvs_data, target_stocks, selected_stocks_ls, stock_adjustments, fig = rank_stock_individuals(stocks_data, 
                                                                                                            top_sectors_mvs_result['corr_matrix'], 
                                                                                                            top_sectors_mvs_result['target_sectors'], 
                                                                                                            d=config['mt_d'], 
                                                                                                            top_mvs_n=config['top_mvs_stock_n'])

    # Market Validation
    market_valid_result = {}
    for ticker in stocks_data.keys():
        stock = stocks_data[ticker]['history']
        data = pd.DataFrame(stock).T
        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"
        data = data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low':'low', 'Volume':'vol'}).drop(columns=['Dividends', 'Stock Splits'])
        
        # Calculate moving averages and indicators
        ma_result = cal_ma(data, 
                           config['hf_d'], 
                           config['st_d'],
                           config['mt_d'],
                           config['lt_d'])

        market_valid_result[ticker] = bool(ma_result['golden_lt']) and bool(ma_result['golden_st']) and bool(ma_result['golden_mt'])



    
    
    save_data = {
        'target_stocks': target_stocks,
        'mvs_result': all_stock_mvs_data,
        'selected_stocks_ls': selected_stocks_ls,
        'stock_adjustments': stock_adjustments,
        'stocks_fig': fig.to_json(),
        'market_valid': market_valid_result
    }

    # Save your desired data into a pickle file
    with open('data/top_stocks_mvs_result.pkl', 'wb') as f:
        pickle.dump(save_data, f)

    add_sys_log('', 'Calculated MVS and saved results.')
    return save_data


@app.route('/load_stock_mvs')
def load_stock_mvs():
    with open('data/top_stocks_mvs_result.pkl', 'rb') as f:
        top_stocks_mvs_result = pickle.load(f)
    return jsonify(top_stocks_mvs_result)



# ==================================== #
#          Build Portfolio  
# ==================================== #   
# Portfolio Page
@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

# Insert Portfolio
@app.route('/build_portfolio', methods=['POST'])
def build_portfolio():
    
    # Get Selected Stocks for Portfolio building
    data = request.get_json()
    selected_stocks_ls = data.get('stocks', [])
    add_sys_log('', 'Insert into Porfolio:')
    add_sys_log('', f"[{', '.join(selected_stocks_ls)}]" )
    print('Selected stocks:', selected_stocks_ls)
    
    with open('data/stocks_data.json', 'r') as f:
        stocks_data_json = f.read()
    stocks_data = pd.read_json(stocks_data_json, orient='records')
    
    with open('data/top_stocks_mvs_result.pkl', 'rb') as f:
        top_stocks_mvs_result = pickle.load(f)    
    
    
    results = {}

    # Loop through all selected stocks
    for ticker in selected_stocks_ls:
        stock = stocks_data[ticker]['history']
        data = pd.DataFrame(stock).T
        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"
        data = data.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low':'low', 'Volume':'vol'}).drop(columns=['Dividends', 'Stock Splits'])

        # Calculate moving averages and indicators
        ma_result = cal_ma(data, config['hf_d'], config['st_d'], config['mt_d'], config['lt_d'])
        
        # Calculate the ATR
        atr_result = cal_atr(data, config['hf_d'])
        
        # Combine MA and ATR results
        results[ticker] = {**ma_result, **atr_result}

        # Plot Price Series
        fig = plot_portfolio_individual(data, ma_result, atr_result, config['lt_d'])
        results[ticker]['fig'] = pio.to_json(fig)

    # Get the 'mvs' for each stock
    mvs_values = {k: top_stocks_mvs_result['mvs_result']['mvs'].get(k, 0) for k in results.keys()}

    # Sort the stocks by 'mvs' and rank them
    sorted_stocks = sorted(mvs_values, key=mvs_values.get, reverse=True)
    ranking = {stock: rank+1 for rank, stock in enumerate(sorted_stocks)}

    # Add the ranks to the results
    for ticker in results.keys():
        results[ticker]['rank'] = ranking[ticker]

    optimized_weights, performance, conf_interval = cal_mpt(stocks_data, selected_stocks_ls, mpt_d=config['lt_d'], hold_d=config['hf_d'])

    # Combine all your results into one dictionary
    figs = {ticker: result['fig'] for ticker, result in results.items()}
    result = {
        'fig': figs, 
        'optimized_weights': optimized_weights, 
        'performance': performance,  
        'confidence_interval': conf_interval, 
    }

    # Save the result into a pickle file
    with open('data/portfolio_result.pkl', 'wb') as f:
        pickle.dump(result, f)
    add_sys_log('', 'Completed and saved.')
    return "Success"


# Optimise Portfolio
@app.route('/cal_portfolio', methods=['POST'])
def cal_portfolio():
    # Get data from the POST request
    data = request.get_json()

    # Extract Portfolio Value and Transaction Cost from Input 
    portfolio_value = int(data.get('portfolioValue'))
    transaction_cost = int(data.get('transactionCost'))

    # Load Portfolio Result 
    with open('data/portfolio_result.pkl', 'rb') as f:
        portfolio_result = pickle.load(f)

    with open('data/stocks_data.json', 'r') as f:
        stocks_data_json = f.read()
    stocks_data = pd.read_json(stocks_data_json, orient='records')

    # Extract Optimized Weights from Portfolio Result
    add_sys_log('', 'Optimising Portfolio.')
    optimized_weights = portfolio_result['optimized_weights']

    # Find Select Stocks with Weights > 0
    selected_stocks = {stock: weight for stock, weight in optimized_weights.items() if weight > 0}

    # Calculate Total Cost
    total_cost = len(selected_stocks) * 2 * transaction_cost

    # Calculate Net Portfolio Value 
    net_portfolio_value = portfolio_value - total_cost
    
    # Finalise Portfolio
    finalised_portfolio = {'individuals': {}, 'portfolio': {}}
    stocks_total_value = 0
    for stock, weight in optimized_weights.items():
        data = {}
        stock_data = stocks_data[stock]['history']
        stock_df = pd.DataFrame(stock_data).T
        stock_df.index = pd.to_datetime(stock_df.index)
        stock_df.index.name = "Date"

        last_close_price = stock_df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low':'low', 'Volume':'vol'}).drop(columns=['Dividends', 'Stock Splits']).iloc[-1]['close']
        data['Last_Close_Price'] = last_close_price
        data['Weight'] = weight if weight > 0 else 0
        data['fig'] = portfolio_result['fig'][stock]

        stock_total_value = 0
        if weight > 0:
            volume = math.floor((weight * net_portfolio_value) / last_close_price)
            stock_total_value = volume * last_close_price
            data['Adjusted_Volume'] = volume
            data['Total_Value'] = stock_total_value
        else:
            data['Adjusted_Volume'] = 0
            data['Total_Value'] = 0
        
        stocks_total_value += stock_total_value
        finalised_portfolio['individuals'][stock] = data

    finalised_portfolio['portfolio']['optimized_weights'] = portfolio_result['optimized_weights']
    finalised_portfolio['portfolio']['performance'] = portfolio_result['performance']
    finalised_portfolio['portfolio']['confidence_interval'] = portfolio_result['confidence_interval']
    finalised_portfolio['portfolio']['portfolio_value'] = stocks_total_value
    finalised_portfolio['portfolio']['cost'] = total_cost
    finalised_portfolio['portfolio']['portfolio_return'] = stocks_total_value * portfolio_result['performance'][0]
   
    # Save the result into a pickle file
    with open('data/finalised_portfolio_result.pkl', 'wb') as f:
        pickle.dump(finalised_portfolio, f)
    add_sys_log('', 'Completed and saved.')   
    
    return jsonify(finalised_portfolio)


@app.route('/load_portfolio')
def load_portfolio():
    with open('data/finalised_portfolio_result.pkl', 'rb') as f:
        finalised_portfolio_result = pickle.load(f)
    return jsonify(finalised_portfolio_result)



if __name__ == '__main__':
    app.run(debug=True)